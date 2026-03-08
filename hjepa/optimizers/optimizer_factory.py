import torch
from hjepa.optimizers.lars import LARS, exclude_bias_and_norm
import enum


class OptimizerType(enum.Enum):
    Adam = "sgd"
    LARS = "lars"


class OptimizerFactory:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_type: str,
        base_lr: float,
        l1_to_l2_lr_ratio: float,
    ):
        self.model = model
        self.optimizer_type = optimizer_type
        self.base_lr = base_lr
        self.l1_to_l2_lr_ratio = l1_to_l2_lr_ratio

    def create_optimizer(self):
        if self.optimizer_type == OptimizerType.LARS:
            optimizer = LARS(
                self.model.parameters(),
                lr=0,
                weight_decay=1e-6,
                weight_decay_filter=exclude_bias_and_norm,
                lars_adaptation_filter=exclude_bias_and_norm,
            )
        elif self.optimizer_type == OptimizerType.Adam:

            models = {
                "level1": self.model.level1,
                "level2": self.model.level2,
            }

            params_list = []

            for level, model in models.items():
                if model is None:
                    continue

                if level == "level1" and self.model.level2 is not None:
                    lr = self.base_lr * self.l1_to_l2_lr_ratio
                else:
                    lr = self.base_lr

                params_list.append(
                    {
                        "params": model.parameters(),
                        "lr": lr,
                    }
                )

                if model.predictor.ensemble_params is not None:
                    params_list.append(
                        {
                            "params": model.predictor.ensemble_params.values(),
                            "lr": lr,
                        }
                    )

            optimizer = torch.optim.Adam(
                params_list,
                weight_decay=1e-6,
            )
        else:
            raise NotImplementedError("Unknown optimizer type")

        return optimizer
