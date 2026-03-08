import unittest

from hjepa.models.jepa import JEPA, JEPAConfig
from objectives.vicreg import VICRegObjectiveConfig, VICRegObjective
from objectives.idm import IDMObjectiveConfig, IDMObjective

from utils import create_mock_batch


class TestObjectives(unittest.TestCase):
    def test_vicreg(self):
        config = JEPAConfig(n_channels=2)
        model = JEPA(config).cuda()

        batch = create_mock_batch()

        input_states = batch.states.transpose(1, 0).cuda()  # swap batch and time
        # swap batch and time, take first dot's actions
        actions = batch.actions.transpose(1, 0)[:, :, 0].cuda()

        # only actions
        result = model.forward_posterior(input_states, actions)

        # without projector
        objective = VICRegObjective(
            VICRegObjectiveConfig(repr_dim=config.repr_dim, projector="id")
        )
        objective(batch, result)

        # with projector
        objective = VICRegObjective(
            VICRegObjectiveConfig(repr_dim=config.repr_dim, projector="128")
        )
        objective(batch, result)

    def test_idm(self):
        config = JEPAConfig(n_channels=2)
        model = JEPA(config).cuda()

        batch = create_mock_batch()

        input_states = batch.states.transpose(1, 0).cuda()  # swap batch and time
        # swap batch and time, take first dot's actions
        actions = batch.actions.transpose(1, 0)[:, :, 0].cuda()

        # only actions
        result = model.forward_posterior(input_states, actions)

        objective = IDMObjective(IDMObjectiveConfig(repr_dim=config.repr_dim))

        objective(batch, result)
