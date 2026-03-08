import torch
from pldm_envs.utils.normalizer import Normalizer


class BaseMPCObjective:
    """Base class for MPC objective.
    This is a callable that takes encodings and returns a tensor -
    objective to be optimized.
    """

    def __call__(self, encodings: torch.Tensor) -> torch.Tensor:
        pass


class ActionChangeObjective(BaseMPCObjective):
    """Objective to ensure smoothness of actions with greater penalty for larger deviations"""

    def __init__(self, alpha: float = 0.5, epsilon: float = 1e-8):
        self.alpha = alpha
        self.epsilon = epsilon

    def __call__(self, actions: torch.Tensor, diff_loss_idx=None):
        """
        actions: B x T x 2
        """
        B, T, _ = actions.shape
        # Adding epsilon to avoid division by zero in norm calculation
        magnitudes = (
            torch.norm(actions, p=2, dim=2) + self.epsilon
        )  # shape (bs, time_steps)

        # Compute squared differences between consecutive magnitudes
        magnitude_diffs = torch.pow(
            torch.diff(magnitudes, dim=1), 2
        )  # shape (bs, time_steps-1)
        # Compute angles using atan2 for each action

        angles = torch.atan2(actions[..., 1], actions[..., 0])  # shape (bs, time_steps)

        # Calculate squared differences between consecutive angles
        angle_diffs = torch.diff(angles, dim=1)  # shape (bs, time_steps-1)
        angle_diffs = (
            torch.remainder(angle_diffs + torch.pi, 2 * torch.pi) - torch.pi
        )  # normalize angle
        angle_diffs = torch.pow(angle_diffs, 2)  # square the absolute value

        if diff_loss_idx is not None:
            mean_magnitude_diffs = torch.zeros(B)
            mean_angle_diffs = torch.zeros(B)
            for i in range(B):
                mean_magnitude_diffs[i] = magnitude_diffs[i, : diff_loss_idx[i]].mean()
                mean_angle_diffs[i] = angle_diffs[i, : diff_loss_idx[i]].mean()
        else:
            # Mean of squared magnitude differences and squared angle differences
            mean_magnitude_diffs = torch.mean(magnitude_diffs, dim=1)  # shape (bs)
            mean_angle_diffs = torch.mean(angle_diffs, dim=1)  # shape (bs)

        # Combine the losses with a weighting factor
        combined_loss = (
            self.alpha * mean_angle_diffs + (1 - self.alpha) * mean_magnitude_diffs
        )

        # Sum across batch
        combined_loss = combined_loss.sum()

        return combined_loss


class SingleStepReprTargetMPCObjective(BaseMPCObjective):
    """Objective to measure the cost to target representation for one time step"""

    def __init__(self, target_enc: torch.Tensor):
        """_summary_
        Args:
            target_enc (D):
        """
        self.target_enc = target_enc

    def __call__(self, state, action):
        """encoding shape is B x D"""
        diff = (state - self.target_enc).pow(2)
        return diff.mean(dim=1)


class PosTargetMPCObjective(BaseMPCObjective):
    """Objective to minimize distance to the target representation."""

    def __init__(
        self,
        target_pos: torch.Tensor,
        prober: torch.nn.Module,
        normalizer: Normalizer,
        sum_all_diffs: bool = False,
        loss_coeff_first: float = 0.1,
        loss_coeff_last: float = 1,
    ):
        self.sum_all_diffs = sum_all_diffs
        self.prober = prober
        self.normalizer = normalizer
        self.normed_target_pos = self.normalizer.normalize_location(target_pos)[
            ..., 0, :
        ]
        self.loss_coeff_first = loss_coeff_first
        self.loss_coeff_last = loss_coeff_last

    def __call__(self, encodings: torch.Tensor) -> torch.Tensor:
        pred_locs = self.prober(encodings)
        """encodings shape is T x B x D"""
        diff = (pred_locs[:, :, 0] - self.normed_target_pos.unsqueeze(0).detach()).pow(
            2
        )
        # diff is of dimension TxBx2 now
        if not self.sum_all_diffs:  # with this flag, we only look at last 3
            diff = diff[-3:]
        diff = diff.mean(dim=0)  # Just take the last three timesteps
        # Replace the line above with diff.mean(dim=0) if you want to use
        # the mean of all timesteps
        # diff is of dimension Bx2 now
        # Now we sum and not average over batch becasue we have
        # separate actions for each batch element.
        # Therefore optimization doesn't really change with batch size.
        return diff.sum(dim=0).mean(dim=0)  # sum batch and avg over coordinates


class EigfObjective(BaseMPCObjective):
    """Objective to minimize distance to the target representation.
    Supports batching.
    """

    def __init__(
        self,
        idx: torch.Tensor,
        minimize: torch.Tensor,
        sum_all_diffs: bool = False,
        discount: float = 1.0,
    ):
        self.idx = idx
        self.sum_all_diffs = sum_all_diffs
        self.signs = minimize.int() * 2 - 1
        self.discount = discount

    def __call__(self, encodings: torch.Tensor, sum_batch: bool = True) -> torch.Tensor:
        """encodings shape is T x B x D"""
        if self.idx.shape[0] == 1:
            # repeat for B batch elements
            indices = self.idx.repeat(encodings.shape[1])
        else:
            indices = self.idx
        encoding_idxs = torch.gather(
            encodings,
            2,
            indices.unsqueeze(0).unsqueeze(-1).repeat(encodings.shape[0], 1, 1),
        )[:, :, 0]

        assert encoding_idxs.shape == encodings.shape[:2]
        assert encoding_idxs[0, 0] == encodings[0, 0, indices[0]]
        diff = encoding_idxs * self.signs.unsqueeze(0)

        if not self.sum_all_diffs:
            diff = diff[-1:]
        # multiply with discount
        diff = diff * torch.pow(
            self.discount,
            torch.arange(diff.shape[0], device=diff.device).float().unsqueeze(1),
        )
        # diff over time
        diff = diff.mean(dim=0)
        if sum_batch:
            diff = diff.sum()
        return diff


class EigfContObjective(BaseMPCObjective):
    """Objective to minimize distance to the target representation.
    Supports batching.
    """

    def __init__(self, coeffs: torch.Tensor, sum_all_diffs: bool = False):
        """Coeffs is of size B x D or D"""
        self.coeffs = coeffs
        if len(self.coeffs.shape) == 1:
            self.coeffs = self.coeffs.unsqueeze(0)
        self.sum_all_diffs = sum_all_diffs

    def __call__(self, encodings: torch.Tensor, sum_batch: bool = True) -> torch.Tensor:
        """encodings shape is T x B x D"""
        # Unsqueeze coeffs for time
        costs = encodings * self.coeffs.unsqueeze(0)
        costs = costs.sum(dim=-1)

        if not self.sum_all_diffs:
            costs = costs[-1:]
        # diff over time
        costs = costs.mean(dim=0)
        if sum_batch:
            costs = costs.sum()
        return costs


# to refactor later
def l2_plan_objective(
    hjepa,
    current_enc,
    action_normalizer,
    actions,
    target_repr,
    sum_all_diffs,
    diff_loss_idx,
    sum_last_n,
):
    batch_size = current_enc.shape[0]

    if action_normalizer is not None:
        actions.data = action_normalizer(actions)

    actions_n = actions

    if hjepa.level2.predictor.prior_model is not None:
        forward_result = hjepa.level2.forward_prior(
            input_states=current_enc,
            actions=actions_n.permute(1, 0, 2),
            repr_input=True,
        )
    else:
        forward_result = hjepa.level2.forward_posterior(
            input_states=current_enc,
            actions=actions_n.permute(1, 0, 2),
            encode_only=False,  # We want predictions, not just encodings
        )

    all_encs = forward_result.pred_output.predictions
    diff = (all_encs - target_repr.unsqueeze(0)).pow(2)
    diff = diff.mean(dim=-1)

    if sum_all_diffs:
        diff = diff.mean(dim=0).sum(dim=0)
    elif diff_loss_idx is None:
        # diff = diff[-1].sum(dim=0)
        diff = diff[-sum_last_n:].mean(dim=0).sum(dim=0)
    else:
        # diff = diff[diff_loss_idx, torch.arange(batch_size)].sum(dim=0)

        new_diff = torch.empty(batch_size)
        for i in range(batch_size):
            start_idx = max(0, diff_loss_idx[i] - sum_last_n + 1)
            end_idx = diff_loss_idx[i] + 1
            new_diff[i] = diff[start_idx:end_idx, i].mean()
        diff = new_diff.sum(dim=0)

    loss = diff
    return loss
