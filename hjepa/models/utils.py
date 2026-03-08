import torch
from torch import nn


def flatten_conv_output(x):
    if len(x.shape) == 4:
        bs, ch, h, w = x.shape
        return x.view(bs, -1)
    elif len(x.shape) == 5:
        t, bs, ch, h, w = x.shape
        return x.view(t, bs, -1)
    elif len(x.shape) > 5:
        raise ValueError(
            f"Expected 3 or 4 or 5 dimensions, but got {len(x.shape)} dimensions."
        )
    else:
        return x


def flatten_ensemble_conv_output(x):
    if len(x.shape) == 5:
        ensembles, bs, ch, h, w = x.shape
        return x.view(ensembles, bs, -1)
    elif len(x.shape) == 6:
        t, ensembles, bs, ch, h, w = x.shape
        return x.view(t, ensembles, bs, -1)
    elif len(x.shape) > 6:
        raise ValueError(
            f"Expected 5 or 6 dimensions, but got {len(x.shape)} dimensions."
        )
    else:
        return x


def get_output_channels(conv_net):
    """
    Get the number of output channels from a convolutional network.
    """
    out_channels = None
    for layer in conv_net.modules():
        if isinstance(layer, torch.nn.Conv2d):
            out_channels = layer.out_channels
    return out_channels


def build_conv(
    layers_config,
    input_dim=None,
    output_dim=None,
    apply_norm=True,
    group_factor=4,
    last_layer_act_norm=False,
):
    input_channels = input_dim[0]
    layers = []
    for i in range(len(layers_config) - 1):
        if isinstance(layers_config[i][0], str) and "pool" in layers_config[i][0]:
            _, kernel_size, stride, padding = layers_config[i]

            if layers_config[i][0] == "avg_pool":
                pool_layer = nn.AvgPool2d(
                    kernel_size=kernel_size, stride=stride, padding=padding
                )
            elif layers_config[i][0] == "max_pool":
                pool_layer = nn.MaxPool2d(
                    kernel_size=kernel_size, stride=stride, padding=padding
                )
            layers.append(pool_layer)
        elif layers_config[i][0] == "pad":
            _, padding = layers_config[i]
            layers.append(nn.ZeroPad2d(padding))
        else:
            in_channels, out_channels, kernel_size, stride, padding = layers_config[i]

            # we override input_channels for first layer with explicit argument
            if i == 0:
                in_channels = input_channels

            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            if apply_norm:
                layers.append(nn.GroupNorm(out_channels // group_factor, out_channels))
            layers.append(nn.ReLU())

    # last layer
    last_layer_config = layers_config[-1]

    if last_layer_config[0] == "fc":
        _, fc_in_dim, fc_out_dim = last_layer_config

        if output_dim is not None:
            fc_out_dim = output_dim

        if fc_in_dim == -1:
            # we need to infer this
            prev_conv_net = nn.Sequential(*layers)
            sample_input = torch.rand(input_dim).unsqueeze(0)
            sample_output = prev_conv_net(sample_input)
            prev_conv_net = None
            fc_in_dim = torch.prod(torch.tensor(sample_output.shape)).item()

        layers.append(nn.Flatten(1, -1))
        layers.append(nn.Linear(fc_in_dim, fc_out_dim))
    elif isinstance(last_layer_config[0], str) and "pool" in last_layer_config[0]:
        _, kernel_size, stride, padding = last_layer_config
        if last_layer_config[0] == "avg_pool":
            pool_layer = nn.AvgPool2d(
                kernel_size=kernel_size, stride=stride, padding=padding
            )
        elif last_layer_config[0] == "max_pool":
            pool_layer = nn.MaxPool2d(
                kernel_size=kernel_size, stride=stride, padding=padding
            )
        layers.append(pool_layer)
    else:
        in_channels, out_channels, kernel_size, stride, padding = last_layer_config
        if output_dim is not None:
            out_channels, _, _ = output_dim

        if not layers:
            # we override input_channels for first layer with explicit argument
            in_channels = input_channels

        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

    if last_layer_act_norm:
        if apply_norm:
            layers.append(nn.GroupNorm(out_channels // group_factor, out_channels))
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class Expander2D(nn.Module):
    """
    This class takes in input of shape (..., n) and expand it into planes (..., n, w, h)
    """

    def __init__(self, w, h):
        super(Expander2D, self).__init__()
        self.w = w
        self.h = h

    def forward(self, x):
        # Reshape to (..., n, 1, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)

        # TODO: TO REMOVE THIS HACK LATER
        # if x.shape[1] > 2:
        #     x = x[:, :2]

        # Calculate the number of dimensions excluding the last 2
        num_dims = x.dim() - 2

        # Create a repeat pattern that matches the number of dimensions
        repeat_pattern = [1] * num_dims + [self.w, self.h]

        # Repeat the last two dimensions to create the (w, h) planes
        x = x.repeat(*repeat_pattern)

        return x
