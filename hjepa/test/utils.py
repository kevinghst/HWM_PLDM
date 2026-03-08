import torch

from hjepa.data.single import Sample


def create_mock_batch():
    bsize = 16
    T = 8
    n_channels = 2
    action_dim = 2
    input_states = torch.zeros(bsize, T, n_channels, 24, 24).cuda()
    actions = torch.zeros(bsize, T - 1, 1, action_dim).cuda()
    locations = torch.randn(bsize, T, 1, 2).cuda()
    bias_angle = torch.randn(bsize, 2).cuda()
    return Sample(
        states=input_states, actions=actions, locations=locations, bias_angle=bias_angle
    )
