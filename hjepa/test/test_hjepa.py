import unittest

import torch

from hjepa.models.hjepa import HJEPA, HJEPAConfig
from hjepa.models.jepa import JEPAConfig


class TestHJEPA(unittest.TestCase):
    def test_forward_posterior(self):
        l1_config = JEPAConfig()
        l2_config = JEPAConfig(
            backbone_arch="", input_dim=l1_config.repr_dim, z_dim=16, action_dim=0
        )
        config = HJEPAConfig(level1=l1_config, level2=l2_config)

        model = HJEPA(config)

        bsize = 16
        T = 8
        input_states = torch.zeros(T, bsize, l1_config.n_channels, 28, 28)
        actions = torch.zeros(T - 1, bsize, l1_config.action_dim)

        result = model.forward_posterior(input_states, actions)
        self.assertEqual(
            result.level1.state_predictions.shape,
            torch.Size([T, bsize, l1_config.repr_dim]),
        )

        self.assertEqual(
            result.level2.state_predictions.shape,
            torch.Size([T // config.step_skip, bsize, l2_config.repr_dim]),
        )

        self.assertEqual(
            result.level2.posteriors.shape,
            torch.Size([T // config.step_skip - 1, bsize, l2_config.z_dim]),
        )

        config.disable_l2 = True
        model = HJEPA(config)
        result = model.forward_posterior(input_states, actions)
        self.assertEqual(
            result.level1.state_predictions.shape,
            torch.Size([T, bsize, l1_config.repr_dim]),
        )
        # forward prior should work when l2 is disabled.
        result = model.forward_prior(input_states[0], actions)
        self.assertEqual(
            result.level1.state_predictions.shape,
            torch.Size([T, bsize, l1_config.repr_dim]),
        )
