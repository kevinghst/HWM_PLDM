from typing import Optional, List, Union

import torch
import numpy as np
from pldm.models.utils import flatten_conv_output

"""
These objectives are used for D4RL environments
"""


class BaseMPCObjective:
    """Base class for MPC objective.
    This is a callable that takes encodings and returns a tensor -
    objective to be optimized.
    """

    def __call__(self, encodings: torch.Tensor) -> torch.Tensor:
        pass

    def set_target(self, target_obs: torch.Tensor):
        pass


class ReprTargetMPCObjective(BaseMPCObjective):
    """Objective to minimize distance to the target representation."""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        sum_all_diffs: bool = False,
        loss_coeff_first: float = 1,
        loss_coeff_last: float = 1,
        target_enc: Optional[torch.Tensor] = None,
        idx: Optional[Union[int, List[int]]] = None,
        pred_encoder: Optional[torch.nn.Module] = None,
        proprio_cost: bool = False,
    ):
        self.sum_all_diffs = sum_all_diffs
        self.model = model
        self.target_enc = target_enc
        self.idx = idx
        self.loss_coeff_first = loss_coeff_first
        self.loss_coeff_last = loss_coeff_last
        self.pred_encoder = pred_encoder
        self.proprio_cost = proprio_cost

    def set_target(self, target_obs: torch.Tensor, repr_input: bool):
        if repr_input:
            from pldm.models.utils import flatten_conv_output

            if len(target_obs.shape) >= 3:
                if len(target_obs.shape) == 3:
                    target_obs = target_obs.unsqueeze(0)
                target_obs = flatten_conv_output(target_obs)
            self.target_enc = target_obs
        else:
            self.target_enc = self.model.backbone(
                target_obs.float().cuda()
            ).encodings.detach()

    def set_idx(self, idx: Union[int, List[int]]):
        self.idx = idx

    def __call__(
        self,
        encodings: torch.Tensor,
        diff_loss_idx: Optional[torch.Tensor] = None,
        idx: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        encodings shape is T x B x D. Including current location.
        self.target_enc shape is B x D
        """
        # TODO: make {sum_last_n} configurable instead of using 3

        # proprio logic
        if self.model.backbone.output_proprio_dim and not self.proprio_cost:
            encodings = self.model.remove_proprio(encodings)

        # we don't diff the current location against target
        encodings = encodings[1:]

        if self.pred_encoder is not None:
            # might need to do this per timestep if OOM
            t, b = encodings.shape[0], encodings.shape[1]
            proper_shape = [t * b] + list(self.pred_encoder.input_dim)
            encodings = encodings.view(proper_shape)
            encodings = self.pred_encoder(encodings)
            encodings = flatten_conv_output(encodings).view(t, b, -1)
        else:
            encodings = flatten_conv_output(encodings)

        target = flatten_conv_output(self.target_enc).unsqueeze(0)
        if self.idx is not None and idx is None:
            idx = self.idx
        if idx is not None:
            target = target[:, idx]
            if isinstance(idx, int):
                target = target.unsqueeze(1)

        diff = (encodings - target).pow(2)

        T = diff.shape[0]
        if self.sum_all_diffs:
            loss_coeffs = np.linspace(self.loss_coeff_first, self.loss_coeff_last, T)
        else:
            if T < 3:
                loss_coeffs = [1] * T
            else:
                loss_coeffs = [0] * T
                loss_coeffs[-3:] = [1, 1, 1]

        loss_coeffs = torch.tensor(
            loss_coeffs, dtype=torch.float32, device=diff.device
        ).view(-1, 1, 1)

        diff = diff * loss_coeffs

        # sum over T and B, mean over dim
        return diff.sum(dim=0).sum(dim=0).mean(dim=0)


class ReprTargetMPCObjective2(BaseMPCObjective):
    """
    A slight variant of ReprTargetMPCObjective.
    Accepts an additional argument 'diff_loss_idx' (bs,)
    will only take the loss up to those indices
    """

    # TODO: Merge this with ReprTargetMPCObjective

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        sum_all_diffs: bool = False,
        target_enc: Optional[torch.Tensor] = None,
        idx: Optional[Union[int, List[int]]] = None,
        sum_last_n: int = 3,
    ):
        self.model = model
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs
        self.sum_last_n = sum_last_n

    def set_target(self, target_obs: torch.Tensor, repr_input: bool):
        if repr_input:
            self.target_enc = target_obs
        else:
            self.target_enc = self.model.encoder(target_obs.float().cuda()).detach()

    def set_idx(self, idx: Union[int, List[int]]):
        self.idx = idx

    def __call__(
        self,
        encodings: torch.Tensor,
        diff_loss_idx: torch.Tensor,
        idx: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        encodings shape is T x B x D
        self.target_enc shape is B x D
        """
        # we don't diff the current location against target
        encodings = encodings[1:]

        target = self.target_enc.unsqueeze(0)
        if self.idx is not None and idx is None:
            idx = self.idx
        if idx is not None:
            target = target[:, idx]
            if isinstance(idx, int):
                target = target.unsqueeze(1)

        # TODO: make {sum_last_n} configurable instead of using 3
        batch_size = target.shape[0]
        diff = (encodings - target).pow(2)
        # mean over dimension
        diff = diff.mean(dim=-1)

        # diff is of shape TxB
        # only last timestep's position is taken into account unless
        # sum_all_diffs is True
        # we sum over batch
        # z_reg is the regularization_loss for the latent actions to push them to 0
        if self.sum_all_diffs:
            diff = diff.mean(dim=0).sum(dim=0)
        elif diff_loss_idx is None:
            # diff = diff[-1].sum(dim=0)
            diff = diff[-self.sum_last_n :].mean(dim=0).sum(dim=0)
        else:
            # diff = diff[diff_loss_idx, torch.arange(batch_size)].sum(dim=0)

            new_diff = torch.empty(batch_size, device=encodings.device)
            for i in range(batch_size):
                start_idx = max(0, diff_loss_idx[i] - self.sum_last_n + 1)
                end_idx = diff_loss_idx[i] + 1
                new_diff[i] = diff[start_idx:end_idx, i].mean()
            diff = new_diff.sum(dim=0)

        return diff


class EigfObjective(BaseMPCObjective):
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

    def __call__(
        self, encodings: torch.Tensor, sum_batch: bool = True, idx: Optional[int] = None
    ) -> torch.Tensor:
        """encodings shape is T x B x D"""
        if idx is None:
            if self.idx.shape[0] == 1:
                # repeat for B batch elements
                indices = self.idx.repeat(encodings.shape[1])
                signs = self.signs.repeat(encodings.shape[1])
            else:
                indices = self.idx
                signs = self.signs
        else:
            indices = self.idx[idx].repeat(encodings.shape[1])
            signs = self.signs[idx].repeat(encodings.shape[1])

        encoding_idxs = torch.gather(
            encodings,
            2,
            indices.unsqueeze(0).unsqueeze(-1).repeat(encodings.shape[0], 1, 1),
        )[:, :, 0]

        assert encoding_idxs.shape == encodings.shape[:2]
        assert encoding_idxs[0, 0] == encodings[0, 0, indices[0]]
        diff = encoding_idxs * signs

        if not self.sum_all_diffs:
            diff = diff[-1:]
        # multiply with discount
        if self.discount != 1.0:
            diff = diff * torch.pow(
                self.discount,
                torch.arange(diff.shape[0], device=diff.device).float().unsqueeze(1),
            )
        # diff over time
        diff = diff.mean(dim=0)
        if sum_batch:
            diff = diff.sum()
        return diff

    def set_target(self, target_obs: torch.Tensor):
        """We don't care about the target in this one."""
