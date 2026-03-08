from enum import auto, Enum
import math


class LRSchedule(Enum):
    Constant = auto()
    Cosine = auto()


class Scheduler:
    def __init__(
        self,
        schedule: str,
        base_lr: float,
        data_loader,
        epochs: int,
        optimizer,
        batch_steps=None,
        batch_size=None,
    ):
        self.schedule = schedule
        self.base_lr = base_lr
        self.data_loader = data_loader
        self.epochs = epochs
        self.optimizer = optimizer

        if batch_size is None:
            self.batch_size = data_loader.config.batch_size
        else:
            self.batch_size = batch_size

        if batch_steps is None:
            self.batch_steps = len(data_loader)
        else:
            self.batch_steps = batch_steps

    def adjust_learning_rate(self, step: int):
        if self.schedule == LRSchedule.Constant:
            return self.base_lr
        else:
            max_steps = self.epochs * self.batch_steps
            warmup_steps = int(0.10 * max_steps)
            for param_group in self.optimizer.param_groups:
                base_lr = (
                    param_group["base_lr"] if "base_lr" in param_group else self.base_lr
                )
                base_lr = base_lr * self.batch_size / 256
                if step < warmup_steps:
                    lr = base_lr * step / warmup_steps
                else:
                    step -= warmup_steps
                    max_steps -= warmup_steps
                    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
                    end_lr = base_lr * 0.001
                    lr = base_lr * q + end_lr * (1 - q)
                param_group["lr"] = lr
            return lr
