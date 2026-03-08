from dataclasses import dataclass
from typing import List

import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd


@torch.no_grad()
def get_cov_std_corr(r: torch.Tensor):
    cov = torch.einsum("bf, be -> ef", r, r) / (r.shape[0] - 1)
    stds = diagonal(cov) ** 0.5
    corr = cov / (torch.einsum("i,j -> ij", stds, stds) + 1e-8)
    avg_corr = (off_diagonal(corr).abs()).mean().item()
    avg_cov = (off_diagonal(cov).abs()).mean().item()
    avg_std = (stds).sum().div(r.shape[1]).item()
    return avg_cov, avg_std, avg_corr


@torch.no_grad()
def get_cropped_std_diff(r: torch.Tensor) -> float:
    r_range = r.max(dim=0).values - r.min(dim=0).values + 1e-8
    r_adjusted = r - r.min(dim=0).values / r_range
    std_diff = (r_adjusted.std(dim=0) - np.sqrt(1 / 12)).abs().mean()
    return std_diff


def off_diagonal(x):
    n, m = x.shape[-2:]
    assert n == m
    return (
        x.flatten(-2)[..., :-1]
        .view(*x.shape[:-2], n - 1, n + 1)[..., :, 1:]
        .flatten(-2)
    )


def diagonal(x):
    n, m = x.shape[-2:]
    assert n == m
    return torch.einsum("...ii->...i", x)


@torch.no_grad()
def analyze_repr(gt: torch.Tensor, pred: torch.Tensor, normalize: bool = True):
    """Normalize tensors, compare their std, similarities and covariance.
    Not used for training, only for diagnostics."""
    gt_unit = gt.float() - gt.mean(dim=0)
    pred_unit = pred.float() - pred.mean(dim=0)

    if normalize:
        gt_unit = F.normalize(gt_unit.float(), dim=-1)
        pred_unit = F.normalize(pred_unit.float(), dim=-1)

    avg_cov_gt, avg_std_gt, avg_corr_gt = get_cov_std_corr(gt_unit)
    avg_cov_pred, avg_std_pred, avg_corr_pred = get_cov_std_corr(pred_unit)

    diff = ((gt - pred).float() ** 2).mean()
    assert len(diff.shape) == 0

    avg_cos_sim = (gt_unit * pred_unit).sum(dim=-1).mean()

    avg_gt_norm = gt.norm(p=2, dim=-1).mean()
    avg_pred_norm = pred.norm(p=2, dim=-1).mean()

    diff_sum = ((gt_unit - pred_unit).float() ** 2).sum()

    avg_std_feature_gt = torch.sqrt(gt_unit.var(dim=-1) + 0.0001).mean()
    avg_std_feature_pred = torch.sqrt(pred_unit.var(dim=-1) + 0.0001).mean()
    avg_std_feature = (avg_std_feature_gt + avg_std_feature_pred) / 2

    feature_std_target = 1 / np.sqrt(gt.shape[-1])
    avg_std_gap_feature_gt = (
        feature_std_target - avg_std_feature_gt
    ) / feature_std_target
    avg_std_gap_feature_pred = (
        feature_std_target - avg_std_feature_pred
    ) / feature_std_target

    batch_std_target = 1 / np.sqrt(gt.shape[0])
    avg_std_gap_batch_gt = (batch_std_target - avg_std_gt) / batch_std_target
    avg_std_gap_batch_pred = (batch_std_target - avg_std_pred) / batch_std_target

    avg_gt_std_diff = get_cropped_std_diff(gt)
    avg_pred_std_diff = get_cropped_std_diff(pred)

    return dict(
        diff=diff.item(),
        diff_sum=diff_sum.item(),
        avg_std_feature=avg_std_feature.item(),
        avg_std_feature_gt=avg_std_feature_gt.item(),
        avg_std_feature_pred=avg_std_feature_pred.item(),
        avg_cos_sim=avg_cos_sim.item(),
        avg_gt_norm=avg_gt_norm.item(),
        avg_pred_norm=avg_pred_norm.item(),
        avg_std_gap_feature_gt=avg_std_gap_feature_gt.item(),
        avg_std_gap_feature_pred=avg_std_gap_feature_pred.item(),
        avg_std_gap_batch_gt=avg_std_gap_batch_gt.item(),
        avg_std_gap_batch_pred=avg_std_gap_batch_pred.item(),
        avg_cov_gt=avg_cov_gt,
        avg_cov_pred=avg_cov_pred,
        avg_std_gt=avg_std_gt,
        avg_std_pred=avg_std_pred,
        avg_corr_gt=avg_corr_gt,
        avg_corr_pred=avg_corr_pred,
        avg_gt_std_diff=avg_gt_std_diff,
        avg_pred_std_diff=avg_pred_std_diff,
        # eigenvals_histogram=svd.detach().cpu().tolist(),
    )


@dataclass
class VICRegDiagnostics:
    """A class to store info for debugging VICReg."""

    std_loss: torch.Tensor
    cov_loss: torch.Tensor
    repr_loss: torch.Tensor
    enc_std_loss: torch.Tensor
    enc_cov_loss: torch.Tensor
    action_loss: torch.Tensor
    reconstruction_loss: torch.Tensor

    enc: torch.Tensor
    pred_enc: List[torch.Tensor]

    proj: torch.Tensor
    pred_proj: List[torch.Tensor]

    def build_log_dict(self, steps=["flat", -1, 0, 1, 3, 7, 15]):
        res = dict(
            std_loss=self.std_loss.item(),
            enc_std_loss=self.enc_std_loss.item(),
            enc_cov_loss=self.enc_cov_loss.item(),
            cov_loss=self.cov_loss.item(),
            repr_loss=self.repr_loss.item(),
            action_loss=self.action_loss.item(),
            reconstruction_loss=self.reconstruction_loss.item(),
        )

        for step in steps:
            if step != "flat" and step >= len(self.pred_enc):
                break

            if step == "flat":
                suffix = "flat"
                c_enc = self.enc[1:].flatten(
                    start_dim=0, end_dim=1
                )  # 0th doesn't have matching in pred
                c_pred_enc = (
                    torch.stack(self.pred_enc)
                    if isinstance(self.pred_enc, list)
                    else self.pred_enc
                ).flatten(start_dim=0, end_dim=1)
                if self.proj is not None:
                    c_proj = self.proj[1:].flatten(start_dim=0, end_dim=1)
                    c_pred_proj = torch.stack(self.pred_proj).flatten(
                        start_dim=0, end_dim=1
                    )
            else:
                if step >= 0:
                    suffix = step + 1
                    c_enc = self.enc[step + 1]
                    c_pred_enc = self.pred_enc[step]
                    if self.proj is not None:
                        c_proj = self.proj[step + 1]
                        c_pred_proj = self.pred_proj[step]
                else:
                    suffix = "0_no_pred"
                    if self.proj is not None:
                        c_pred_enc = c_enc = self.enc[0]
                        c_pred_proj = c_proj = self.proj[0]

            res[f"{suffix}/enc_analysis"] = analyze_repr(
                c_enc, c_pred_enc, normalize=True
            )
            res[f"{suffix}/enc_analysis_unnormed"] = analyze_repr(
                c_enc, c_pred_enc, normalize=False
            )
            if self.proj is not None:
                res[f"{suffix}/proj_analysis"] = analyze_repr(
                    c_proj, c_pred_proj, normalize=True
                )
                res[f"{suffix}/proj_analysis_unnormed"] = analyze_repr(
                    c_proj, c_pred_proj, normalize=False
                )

        df = pd.io.json.json_normalize(dict(analysis=res), sep="/")
        return df.to_dict(orient="records")[0]
