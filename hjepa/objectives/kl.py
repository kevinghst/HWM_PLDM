from dataclasses import dataclass
from typing import NamedTuple, List, Optional

import torch
import torch.distributions as td
import torch.nn.functional as F

from hjepa.configs import ConfigBase
from hjepa.models.jepa import ForwardResult
from hjepa.models.predictors.enums import PredictorOutput


class KLLossInfo(NamedTuple):
    total_loss: torch.Tensor
    kl_loss: torch.Tensor
    loss_name: str = "kl"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_loss": self.kl_loss.item(),
        }


@dataclass
class KLObjectiveConfig(ConfigBase):
    global_coeff: float = 1.0
    global_prior: bool = True
    discrete: bool = False
    discrete_diversity: bool = False


def calc_kl_continuous(
    uniform_prior: True,
    posteriors: torch.Tensor,
    posterior_mus: torch.Tensor,
    posterior_vars: torch.Tensor,
    prior_mus: Optional[torch.Tensor] = None,
    prior_vars: Optional[torch.Tensor] = None,
    reduction: bool = True,
):
    if uniform_prior:
        prior_mus = torch.zeros_like(posterior_mus)
        prior_vars = torch.ones_like(posterior_vars)

    # we use the prior mu and sigma in result.
    posterior_d = torch.distributions.Normal(posterior_mus, posterior_vars)
    prior_d = torch.distributions.Normal(prior_mus, prior_vars)
    loss = torch.distributions.kl.kl_divergence(posterior_d, prior_d)

    if reduction:
        loss = loss.mean()

    return loss


class KLObjective(torch.nn.Module):
    """Objective to regularize the latent variable
    for the second level JEPA.
    """

    def __init__(self, config: KLObjectiveConfig, name_prefix: str = ""):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix

    def _calc_kl_discrete_diversity(self, result: PredictorOutput):
        posterior_logits = result.posterior_logits
        seq_len, bs, dists, bins = posterior_logits.shape
        posterior_logits = posterior_logits.view(seq_len * dists, bs, bins)

        # Compute the probabilities and log-probabilities from the logits
        probs = F.softmax(posterior_logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)

        # Compute the pairwise KL divergence using broadcasting
        # Shape: (m, bs, bs, n)
        kl_div = probs.unsqueeze(2) * (log_probs.unsqueeze(2) - log_probs.unsqueeze(1))

        # Sum over the last dimension (n) to get the KL divergence for each pair (i, j) in each m
        # Shape: (m, bs, bs)
        kl_div = kl_div.sum(dim=-1)

        # We want the mean of all pairwise KL divergences, excluding the diagonal, for each m
        kl_div_sum = kl_div.sum(dim=(1, 2)) - kl_div.diagonal(dim1=1, dim2=2).sum(
            dim=-1
        )
        diversity_reg = kl_div_sum / (bs * (bs - 1))

        # Take the average over the first dimension (m)
        diversity_reg_mean = -diversity_reg.mean()

        return diversity_reg_mean

    def _calc_kl_discrete(self, result: PredictorOutput):
        posterior_logits = result.posterior_logits
        prior_logits = result.prior_logits
        seq_len, bs, dists, bins = posterior_logits.shape

        posterior_logits = posterior_logits.contiguous().view(bs * seq_len, dists, bins)
        prior_logits = prior_logits.contiguous().view(bs * seq_len, dists, bins)

        kls = torch.distributions.kl.kl_divergence(
            td.Independent(
                td.OneHotCategoricalStraightThrough(logits=posterior_logits), 1
            ),
            td.Independent(
                td.OneHotCategoricalStraightThrough(logits=prior_logits.detach()), 1
            ),
        )

        loss = kls.mean()
        return loss

    def _calc_kl_continuous(self, result: PredictorOutput):
        return calc_kl_continuous(
            uniform_prior=self.config.global_prior,
            posteriors=result.posteriors,
            posterior_mus=result.posterior_mus,
            posterior_vars=result.posterior_vars,
            prior_mus=result.prior_mus,
            prior_vars=result.prior_vars,
        )

    def __call__(self, batch, results: List[ForwardResult]) -> KLLossInfo:
        # results is the list of forward results from jepa levels,
        # from first to last
        result = results[-1]

        if self.config.discrete:
            if self.config.discrete_diversity:
                loss = self._calc_kl_discrete_diversity(result.pred_output)
            else:
                loss = self._calc_kl_discrete(result.pred_output)
        else:
            loss = self._calc_kl_continuous(result.pred_output)

        return KLLossInfo(
            total_loss=loss * self.config.global_coeff,
            kl_loss=loss,
            name_prefix=self.name_prefix,
        )
