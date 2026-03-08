# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from pldm.models.encoders.enums import BackboneOutput
from pldm.models.encoders.base_class import SequenceBackbone


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockID(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        last_activation="relu",
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        last_activation="relu",
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(min(32, planes // 4), planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(min(32, planes // 4), planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        last_activation="relu",
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(min(32, width // 4), width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(min(32, width // 4), width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(
            min(32, planes * self.expansion // 4), planes * self.expansion
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if last_activation == "relu":
            self.last_activation = nn.ReLU(inplace=True)
        elif last_activation == "none":
            self.last_activation = nn.Identity()
        elif last_activation == "sigmoid":
            self.last_activation = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.last_activation(out)

        return out


class ResNet(SequenceBackbone):
    def __init__(
        self,
        block,
        layers,
        filters,
        strides,
        num_channels=3,
        zero_init_residual=False,
        groups=1,
        widen=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_activation="relu",
        inital_maxpool=True,
        final_pool_type="avg_pool",
        final_out_filters=32,
        spatial_output=False,
        expand_factor=2,
        initial_padding=True,
        final_ln=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        self._norm_layer = norm_layer
        self.spatial_output = spatial_output
        # self._last_activation = last_activation

        if initial_padding:
            self.padding = nn.ConstantPad2d(1, 0.0)
        else:
            self.padding = None

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = filters[0]
        self.conv1 = nn.Conv2d(
            num_channels,
            filters[0],
            kernel_size=7,
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn1 = norm_layer(min(32, num_out_filters // 4), num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        if inital_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = None
        self.layer1 = self._make_layer(
            block, num_out_filters, layers[0], stride=strides[0]
        )

        if len(layers) >= 2:
            num_out_filters = filters[1]
            self.layer2 = self._make_layer(
                block,
                filters[1],
                layers[1],
                stride=strides[1],
                dilate=replace_stride_with_dilation[0],
            )
        else:
            self.layer2 = None

        if len(layers) >= 3:
            num_out_filters = filters[2]
            self.layer3 = self._make_layer(
                block,
                filters[2],
                layers[2],
                stride=strides[2],
                dilate=replace_stride_with_dilation[1],
            )
        else:
            self.layer3 = None

        if len(layers) == 4:
            num_out_filters = filters[3]
            self.layer4 = self._make_layer(
                block,
                filters[3],
                layers[3],
                stride=strides[3],
                dilate=replace_stride_with_dilation[2],
                last_activation=last_activation,
            )
        else:
            self.layer4 = None

        if final_pool_type == "avg_pool":
            self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif final_pool_type == "1x1_pool":
            self.final_pool = nn.Conv2d(num_out_filters, final_out_filters, 1, 1, 0)
        elif final_pool_type == "id":
            self.final_pool = None

        self.final_ln = nn.LayerNorm(filters[-1]) if final_ln else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and
        # each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        last_activation="relu",
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(
                    min(planes * block.expansion // 4, 32), planes * block.expansion
                ),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                last_activation=(last_activation if blocks == 1 else "relu"),
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    last_activation=(last_activation if i == blocks - 1 else "relu"),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.padding is not None:
            x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layer1(x)

        if self.layer2 is not None:
            x = self.layer2(x)

        if self.layer3 is not None:
            x = self.layer3(x)

        if self.layer4 is not None:
            x = self.layer4(x)

        if self.final_pool is not None:
            x = self.final_pool(x)

        if not self.spatial_output:
            x = torch.flatten(x, 1)

        x = BackboneOutput(encodings=x)
        return x


def resnet18s_a(**kwargs):
    return (
        ResNet(
            BasicBlock,
            [2, 2],
            inital_maxpool=False,
            spatial_output=True,
            initial_padding=False,
            expand_factor=1,
            final_pool_type="id",
            **kwargs,
        ),
        None,
    )


def resnet18s_b(**kwargs):
    return (
        ResNet(
            BasicBlock,
            [2, 2],
            inital_maxpool=False,
            spatial_output=True,
            initial_padding=False,
            expand_factor=1,
            final_pool_type="1x1_pool",
            final_out_filters=32,
            **kwargs,
        ),
        None,
    )


def resnet18s_c(**kwargs):
    return (
        ResNet(
            BasicBlock,
            [2, 2],
            inital_maxpool=False,
            spatial_output=True,
            initial_padding=False,
            expand_factor=1,
            final_pool_type="1x1_pool",
            final_out_filters=48,
            **kwargs,
        ),
        None,
    )


def resnet18s_d(**kwargs):
    return (
        ResNet(
            BasicBlock,
            [2, 2],
            inital_maxpool=False,
            spatial_output=True,
            initial_padding=False,
            expand_factor=2,
            final_pool_type="1x1_pool",
            final_out_filters=64,
            **kwargs,
        ),
        None,
    )


def resnet18s_e(**kwargs):
    return (
        ResNet(
            BasicBlock,
            [2, 2],
            inital_maxpool=False,
            spatial_output=True,
            initial_padding=False,
            expand_factor=2,
            final_pool_type="1x1_pool",
            final_out_filters=48,
            **kwargs,
        ),
        None,
    )


def resnet18s_f(**kwargs):
    return (
        ResNet(
            BasicBlock,
            [2, 2],
            inital_maxpool=False,
            spatial_output=True,
            initial_padding=False,
            expand_factor=2,
            final_pool_type="1x1_pool",
            final_out_filters=32,
            **kwargs,
        ),
        None,
    )


def resnet18s_g(**kwargs):
    return (
        ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            filters=[64, 128, 128, 128],
            strides=[1, 2, 1, 1],
            num_channels=6,
            zero_init_residual=True,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            last_activation="relu",
            inital_maxpool=False,
            final_pool_type="1x1_pool",
            final_out_filters=64,
            spatial_output=True,
            expand_factor=2,
            initial_padding=True,
        ),
        None,
    )


def resnet18(**kwargs):
    return (
        ResNet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            filters=[64, 128, 256, 512],
            strides=[1, 2, 2, 2],
            **kwargs,
        ),
        512,
    )


def resnet18ID(**kwargs):
    return ResNet(BasicBlockID, [2, 2, 2, 2], **kwargs), 512


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs), 512


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs), 2048


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs), 2048


def resnet50x2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs), 4096


def resnet50x4(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs), 8192


def resnet50x5(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs), 10240


def resnet200x2(**kwargs):
    return ResNet(Bottleneck, [3, 24, 36, 3], widen=2, **kwargs), 4096
