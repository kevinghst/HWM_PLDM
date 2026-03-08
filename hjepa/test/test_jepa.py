import unittest

import torch

from hjepa.models.jepa import JEPA, JEPAConfig


class TestJEPA(unittest.TestCase):
    def test_forward_prior(self):
        config = JEPAConfig()
        model = JEPA(config)

        bsize = 16
        T = 8
        input_states = torch.zeros(bsize, config.n_channels, 24, 24)
        actions = torch.zeros(T - 1, bsize, config.action_dim)

        # only actions
        result = model.forward_prior(input_states, actions)
        self.assertEqual(
            result.state_predictions.shape, torch.Size([T, bsize, config.repr_dim])
        )

        # only latents
        model = JEPA(JEPAConfig(action_dim=0, z_dim=32))
        result = model.forward_prior(input_states, T=T - 1)
        self.assertEqual(
            result.state_predictions.shape, torch.Size([T, bsize, config.repr_dim])
        )

        # both
        model = JEPA(JEPAConfig(action_dim=2, z_dim=32))
        result = model.forward_prior(input_states, actions)
        self.assertEqual(
            result.state_predictions.shape, torch.Size([T, bsize, config.repr_dim])
        )

    def test_forward_posterior(self):
        config = JEPAConfig()

        bsize = 16
        T = 8
        input_states = torch.zeros(T, bsize, config.n_channels, 24, 24)
        actions = torch.zeros(T - 1, bsize, config.action_dim)

        # only actions
        config.z_dim = 0
        config.action_dim = 2
        model = JEPA(config)
        result = model.forward_posterior(input_states, actions)
        self.assertEqual(
            result.state_predictions.shape, torch.Size([T, bsize, config.repr_dim])
        )

        # only latents
        config.z_dim = 2
        config.action_dim = 0
        model = JEPA(config)
        result = model.forward_posterior(input_states, actions=None)
        self.assertEqual(
            result.state_predictions.shape, torch.Size([T, bsize, config.repr_dim])
        )

        # both
        config.z_dim = 2
        config.action_dim = 2
        model = JEPA(config)
        result = model.forward_posterior(input_states, actions)
        self.assertEqual(
            result.state_predictions.shape, torch.Size([T, bsize, config.repr_dim])
        )
