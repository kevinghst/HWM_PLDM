import unittest


from train import TrainConfig, Trainer

from hjepa.data import DatasetType
from objectives import ObjectiveType


class TestIntegration(unittest.TestCase):
    def test_one_level(self):
        config = TrainConfig(quick_debug=True, wandb=False)
        config.hjepa.disable_l2 = True
        config.dataset_type = DatasetType.Wall
        config.hjepa.level1.n_channels = 2
        config.objectives_l1.objectives = [
            ObjectiveType.VICReg,
            ObjectiveType.IDM,
            ObjectiveType.Laplacian,
        ]  # testing all of them at once
        trainer = Trainer(config)
        trainer.train()

    def test_two_levels(self):
        config = TrainConfig(quick_debug=True, wandb=False)
        config.hjepa.disable_l2 = False
        config.dataset_type = DatasetType.Wall
        config.hjepa.level1.n_channels = 2
        config.hjepa.level2.input_dim = config.hjepa.level1.repr_dim
        config.hjepa.level2.backbone_arch = ""
        config.hjepa.level2.action_dim = 0
        config.hjepa.level2.z_dim = 16

        config.objectives_l1.objectives = [
            ObjectiveType.VICReg,
            ObjectiveType.IDM,
            ObjectiveType.Laplacian,
        ]  # Testing all of them at once
        config.objectives_l2.objectives = [
            ObjectiveType.VICReg,
            ObjectiveType.Laplacian,
        ]  # Testing all of them at once, IDM cannot be used at the second level.
        trainer = Trainer(config)
        trainer.train()
