import torch.nn as nn
import torch.nn.functional as F
from pldm.models.encoders.enums import BackboneOutput
from pldm.models.encoders.base_class import SequenceBackbone


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
        self, hidden_dims, activation=nn.GELU, activate_final=False, layer_norm=False
    ):
        super(MLP, self).__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation()
        self.activate_final = activate_final
        self.layer_norm = layer_norm

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i + 1 < len(hidden_dims) or activate_final:
                layers.append(self.activation)
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden_dims[i + 1]))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResnetBlock(nn.Module):
    """ResNet Block."""

    def __init__(self, num_features):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super(ResnetStack, self).__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool = nn.Identity()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(SequenceBackbone):
    """IMPALA encoder."""

    def __init__(
        self,
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        mlp_hidden_dims=(512,),
        layer_norm=False,
        input_channels=2,
        final_ln=False,
    ):
        super(ImpalaEncoder, self).__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm

        input_channels = [input_channels] + list(stack_sizes)

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=input_channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()
        self.mlp = nn.LazyLinear(out_features=512)

        if final_ln:
            self.final_ln = nn.LayerNorm(512)
        else:
            self.final_ln = nn.Identity()

    def forward(self, x, train=True, **kwargs):
        # x = x.float() / 255.0  # Normalize input
        conv_out = x

        for i, stack_block in enumerate(self.stack_blocks):
            conv_out = stack_block(conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out)

        conv_out = F.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm(conv_out.size()[1:])(conv_out)

        out = conv_out.view(conv_out.size(0), -1)
        out = self.mlp(out)
        out = self.final_ln(out)

        out = BackboneOutput(encodings=out)

        return out


# encoder = ImpalaEncoder(final_ln=True)

# test_input = torch.randn(1, 2, 64, 64)

# print(encoder(test_input).shape)
