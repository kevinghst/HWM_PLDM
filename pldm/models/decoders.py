from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from pldm.models.encoders.encoders import ResizeConv2d


class VAEDecoder(torch.nn.Module):
    def __init__(self, embedding_size=512, output_channels=1):
        super().__init__()
        self.k1, self.k2, self.k3, self.k4 = (
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
        )  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        )  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        )  # 2d padding

        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=self.k4,
                stride=self.s4,
                padding=self.pd4,
            ),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=self.k3,
                stride=self.s3,
                padding=self.pd3,
            ),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=8,
                kernel_size=self.k2,
                stride=self.s2,
                padding=self.pd2,
            ),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.Sigmoid(),  # y = (y1, y2, y3) \in [0 ,1]^3
            nn.Conv2d(8, out_channels=output_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.fc1 = nn.Linear(2 * embedding_size, embedding_size)

    def forward(self, z, belief):
        z = self.fc1(torch.cat([z, belief], dim=1))
        x = z.view(-1, 32, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x) * 20
        x = F.interpolate(x, size=(28, 28), mode="bilinear")
        return x


class VAEDecoder_vc(torch.nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.k1, self.k2, self.k3, self.k4 = (
            (3, 3),
            (3, 3),
            (3, 3),
            (3, 3),
        )  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        )  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        )  # 2d padding

        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=self.k4,
                stride=self.s4,
                padding=self.pd4,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16, momentum=0.01),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=self.k3,
                stride=self.s3,
                padding=self.pd3,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8, momentum=0.01),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=8,
                kernel_size=self.k2,
                stride=self.s2,
                padding=self.pd2,
            ),
            nn.ReLU(),  # y = (y1, y2, y3) \in [0 ,1]^3
            nn.BatchNorm2d(8, momentum=0.01),
            nn.Conv2d(8, out_channels=1, kernel_size=3, padding=1),
        )

    #         self.fc1 = nn.Linear(embedding_size, embedding_size)

    def forward(self, z):
        x = z.view(-1, 32, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(28, 28), mode="bilinear")
        return x


class MeNet5Decoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        z_dim: int,
        output_channels: int = 1,
        width_factor: int = 1,
    ):
        super().__init__()
        self.width_factor = width_factor
        self.layers = nn.Sequential(
            ResizeConv2d(
                32 * width_factor,
                32 * width_factor,
                kernel_size=3,
                scale_factor=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32 * width_factor),
            ResizeConv2d(
                32 * width_factor,
                16 * width_factor,
                kernel_size=5,
                scale_factor=3,
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16 * width_factor),
            ResizeConv2d(
                16 * width_factor,
                output_channels,
                kernel_size=5,
                scale_factor=3,
                padding=2,
            ),
        )
        self.fc = nn.Linear(embedding_size + z_dim, 32 * 3 * 3 * self.width_factor)

    def forward(self, x: torch.Tensor, belief: Optional[torch.Tensor] = None):
        if belief is not None:
            x = torch.cat([x, belief], dim=1)
        x = self.fc(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1, 3, 3)
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # 6 by 6, undo avg pool
        x = self.layers(x)
        x = F.interpolate(x, size=(28, 28), mode="bilinear")  # 27 by 27 to 28 by 28
        return x


class MLPDecoder(nn.Module):
    """Decoder for the second level of jepa.
    Here we decode the output of the first level.
    """

    def __init__():
        pass

    def forward(self, x: torch.Tensor):
        pass
