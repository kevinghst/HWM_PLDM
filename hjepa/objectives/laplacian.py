from dataclasses import dataclass
from typing import NamedTuple, List, Optional

import torch
import numpy as np

from hjepa.configs import ConfigBase
from hjepa.models.jepa import ForwardResult


class LaplacianLossInfo(NamedTuple):
    total_loss: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor
    predictor_loss: torch.Tensor
    loss_name: str = "laplacian"
    name_prefix: str = ""
    priorities: Optional[torch.Tensor] = None

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_positive_loss": self.positive_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_negative_loss": self.negative_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_predictor_loss": self.predictor_loss.item(),
        }


@dataclass
class LaplacianObjectiveConfig(ConfigBase):
    random_projector: bool = False
    negative_coeff: float = 1.0
    projector: Optional[str] = None  # deprecated
    c: float = 1.0
    reg: float = 0.0
    discount: float = 0.9
    p1_correction: bool = False
    p2_correction: bool = False

    global_coeff: float = 1.0

    split_batch: bool = False

    predictor_coeff: float = 0.0
    detach_predictor: bool = True

    # This controls whether we add coefficients that guarantee that the
    # eigenvalues are ordered and exact.
    generalized: bool = False
    # Controls whether the negative component of the loss uses the
    # generalized objective or not.
    generalized_negative: bool = False
    remove_d: bool = False

    negative_one_step: bool = False

    generalized_coeff_bias: float = 0
    generalized_coeff_curve: float = 0


def discounted_sampling(ranges, discount):
    """Draw samples from the discounted distribution over 0, ...., n - 1,
    where n is a range. The input ranges is a batch of such n`s.
    The discounted distribution is defined as
    p(y = i) = (1 - discount) * discount^i / (1 - discount^n).
    This function implement inverse sampling. We first draw
    seeds from uniform[0, 1) then pass them through the inverse cdf
    floor[ log(1 - (1 - discount^n) * seeds) / log(discount) ]
    to get the samples.
    from : https://github.com/yifan12wu/rl-laplacian/blob/eb1dec8acae6e48a5a45c2035fa8d02347503235/rl_lap/agent/episodic_replay_buffer.py#L11 # noqa
    """
    assert np.min(ranges) >= 1
    assert discount >= 0 and discount <= 1
    seeds = np.random.uniform(size=ranges.shape)
    if discount == 0:
        samples = np.zeros_like(seeds, dtype=np.int64)
    elif discount == 1:
        samples = np.floor(seeds * ranges).astype(np.int64)
    else:
        samples = np.log(1 - (1 - np.power(discount, ranges)) * seeds) / np.log(
            discount
        )
        samples = np.floor(samples).astype(np.int64)
    return torch.from_numpy(samples)


class LaplacianObjective(torch.nn.Module):
    def __init__(
        self, config: LaplacianObjectiveConfig, repr_dim: int, name_prefix: str = ""
    ):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix

        if self.config.generalized:
            diffs = (
                torch.ones(repr_dim, device=torch.device("cuda"))
                + torch.arange(0, repr_dim, 1, device=torch.device("cuda"))
                * self.config.generalized_coeff_curve
            )
            diffs = (diffs / diffs.sum()) * repr_dim
            self.generalized_coeff = (
                diffs.cumsum(dim=0) + self.config.generalized_coeff_bias
            )
            self.generalized_coeff /= self.generalized_coeff.sum()
            self.generalized_coeff *= repr_dim
            self.generalized_coeff = self.generalized_coeff.flip(dims=(0,))

            self.generalized_coeff_diffs = (
                self.generalized_coeff[:-1] - self.generalized_coeff[1:]
            )
            self.generalized_coeff_diffs = torch.cat(
                [self.generalized_coeff_diffs, self.generalized_coeff[-1:]]
            )
            # we want it to sum to dim, so that the loss is invariant to the dimension

    def predictor_loss(self, encodings: torch.Tensor, predictions: torch.Tensor):
        predictor_loss = torch.tensor(0.0).to(encodings.device)
        if self.config.predictor_coeff > 0.0:
            targets = encodings
            if self.config.detach_predictor:
                targets = targets.detach()
            predictor_loss = (predictions - targets).pow(2).mean(dim=-1).mean()
        return predictor_loss

    def laplacian_loss(self, encodings):
        batch_size = encodings.shape[1]
        if self.config.split_batch:
            # first half is for positive pairs
            positive_encodings = encodings[:, : batch_size // 2]
            # second half is for negative pairs
            negative_encodings = encodings[:, batch_size // 2 :]
        else:
            positive_encodings = encodings
            negative_encodings = encodings

        ranges = np.ones((positive_encodings.shape[1],)) * (
            positive_encodings.shape[0] - 1
        )  # we can shift by T frames

        shifts = (
            discounted_sampling(ranges, self.config.discount).to(
                positive_encodings.device
            )
            + 1
        )

        shifts = shifts.view(1, -1, 1)
        shifts = shifts.repeat(1, 1, positive_encodings.shape[-1])
        targets = torch.gather(positive_encodings, 0, shifts)[0]

        positive_loss = (positive_encodings[0] - targets).pow(2)
        if self.config.generalized:
            # here we multiply the last dimension with range(dim)
            positive_loss = positive_loss * self.generalized_coeff
        positive_loss = positive_loss.mean(dim=-1)

        # positive loss is of shape B
        # average over batch
        assert len(positive_loss.shape) == 1

        # negative component tries to separate the features of random frames
        negative_loss = torch.tensor(0.0).to(encodings.device)
        if self.config.negative_coeff > 0:
            for i in range(negative_encodings.shape[0]):
                # sample a random frame from the batch
                if self.config.generalized_negative:
                    negative_loss += self.generalized_neg_loss(
                        negative_encodings[i], c=self.config.c, reg=self.config.reg
                    )
                else:
                    negative_loss += self.neg_loss(
                        negative_encodings[i], c=self.config.c, reg=self.config.reg
                    )
                if self.config.negative_one_step:
                    break

            # average over time
            if not self.config.negative_one_step:
                negative_loss /= negative_encodings.shape[0]

        return positive_loss, negative_loss

    def __call__(self, _batch, results: List[ForwardResult]) -> LaplacianLossInfo:
        # Note: batch tensors has shape BxT... , result tensors have shape TxB ...

        # positive component just brings together the features of consecutive frames

        # this is by how much we shift in time for positive pairs
        result = results[-1]
        per_element_positive_loss, negative_loss = self.laplacian_loss(
            result.backbone_output.encodings
        )
        positive_loss = per_element_positive_loss.mean()

        predictor_loss = self.predictor_loss(
            result.backbone_output.encodings, result.pred_output.predictions
        )

        # predictor loss
        total_loss = self.config.global_coeff * (
            positive_loss.mean()
            + self.config.negative_coeff * negative_loss.mean()
            + self.config.predictor_coeff * predictor_loss
        )

        return LaplacianLossInfo(
            total_loss=total_loss,
            positive_loss=positive_loss,
            negative_loss=negative_loss,
            predictor_loss=predictor_loss,
            priorities=per_element_positive_loss,
            name_prefix=self.name_prefix,
        )

    def neg_loss(self, x, c=1.0, reg=0.0):
        """
        x: n * d.
        sample based approximation for
        VICREG: x^T x to I
        (E[x x^T] - c * I / d)^2
            = E[(x^T y)^2] - 2c E[x^T x] / d + c^2 / d
        #
        An optional regularization of
        reg * E[(x^T x - c)^2] / n
            = reg * E[(x^T x)^2 - 2c x^T x + c^2] / n
        for reg in [0, 1]
        """
        n = x.shape[0]
        if self.config.remove_d:
            d = 1
        else:
            d = x.shape[1]

        inprods = x @ x.T
        norms = inprods[torch.arange(n), torch.arange(n)]
        if self.config.p1_correction:
            part1 = inprods.pow(2).mean()
        else:
            part1 = inprods.pow(2).sum() - norms.pow(2).sum()
            part1 = part1 / ((n - 1) * n)

        part2 = -2 * c * norms.mean() / d
        part3 = c * c / d
        # regularization
        if reg > 0.0:
            reg_part1 = norms.pow(2).mean()
            reg_part2 = -2 * c * norms.mean()
            reg_part3 = c * c
            reg_part = (reg_part1 + reg_part2 + reg_part3) / n
        else:
            reg_part = 0.0
        return part1 + part2 + part3 + reg * reg_part

    def generalized_neg_loss(self, x, c=1.0, reg=0.0):
        """
        x: n * d.
        sample based approximation for
        VICREG: x^T x to I
        (E[x x^T] - c * I / d)^2
            = E[(x^T y)^2] - 2c E[x^T x] / d + c^2 / d
        #
        An optional regularization of
        reg * E[(x^T x - c)^2] / n
            = reg * E[(x^T x)^2 - 2c x^T x + c^2] / n
        for reg in [0, 1]
        """
        n = x.shape[0]

        inprods = torch.einsum("nd,md->nmd", x, x).cumsum(dim=-1)
        # inprods is of shape N x N x D now
        # last D is inner prods for each prefix
        norms = inprods[torch.arange(n), torch.arange(n)]

        if self.config.p1_correction:
            part1 = inprods.pow(2).mean(dim=[0, 1])
        else:
            part1 = inprods.pow(2).sum(dim=[0, 1]) - norms.pow(2).sum(dim=0)
            part1 = part1 / ((n - 1) * n)

        # part 1 is of dimension d

        # norms shape is N x D
        part2 = -2 * c * norms.sum(dim=0) / (n * n)

        pre_coeff = part1 + part2
        return (pre_coeff * self.generalized_coeff_diffs).mean()
