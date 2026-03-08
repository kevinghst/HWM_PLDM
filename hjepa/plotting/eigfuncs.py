from typing import Mapping, Tuple, Any, List

import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from hjepa.envs.wall import DotWall
from environments.utils.normalizer import Normalizer
from hjepa.data.wall.wall_eigenfunc import WallEigenfunctionsSample
from hjepa.models.jepa import JEPA
from hjepa.logger import Logger
from hjepa.planning.wall.mpc import MPCReport
from hjepa.planning.wall.plotting import default_plot_idxs

IMG_SIZE = 28


def get_dim_grid(m: Mapping[Tuple[int, int], torch.Tensor], dim: int):
    r = torch.zeros(21, 21).cuda()
    for i in range(3, 24):
        for j in range(3, 24):
            r[j - 3, i - 3] = (m[(i, j)][0])[dim]
    return r


def get_product_grid(m: Mapping[Tuple[int, int], torch.Tensor], x: torch.Tensor):
    r = torch.zeros(21, 21).cuda()
    for i in range(3, 24):
        for j in range(3, 24):
            r[j - 3, i - 3] = (m[(i, j)][0]).dot(x)
    return r


def plot_dim(
    ax: Any, dim: int, m: Mapping[Tuple[int, int], torch.Tensor], *args, **kwargs
):
    r = get_dim_grid(m, dim)
    im = ax.imshow(r.detach().cpu(), *args, **kwargs)
    ax.axis("off")
    return im


def plot_product(
    ax: Any,
    m: Mapping[Tuple[int, int], torch.Tensor],
    x: torch.Tensor,
    *args,
    **kwargs,
):
    r = get_product_grid(m, x)
    im = ax.imshow(r.detach().cpu(), *args, **kwargs)
    ax.axis("off")
    return im


def get_repr_grid(jepa: JEPA, normalizer: Normalizer):
    env = DotWall()
    current_state = env.reset()

    current_state_normalized = normalizer.normalize_state(current_state.unsqueeze(0))

    wall_img = env._render_walls(torch.tensor(14).cuda(), torch.tensor(14).cuda())
    # plt.imshow(dot_img.cpu() + wall_img.cpu() * dot_img.max().cpu())

    # build a map of representations at all locations for nearest neighbor heatmap
    m = {}
    vmax = -1e10
    vmin = 1e10
    for i in range(3, 24):
        for j in range(3, 24):
            dot_position = torch.tensor([i, j]).float().cuda()
            dot_img = env._render_dot(dot_position)
            obs = torch.stack([dot_img, wall_img * dot_img.max()], dim=0)
            current_state_normalized = normalizer.normalize_state(obs.unsqueeze(0))
            enc = jepa.backbone(current_state_normalized.cuda())
            m[(i, j)] = enc
            vmax = max(enc.max(), vmax)
            vmin = min(enc.min(), vmin)

    return m, vmin, vmax


def plot_eigfuncs(
    jepa: JEPA,
    normalizer: Normalizer,
    notebook: bool = False,
    n_eig: int = 20,
):
    m, vmin, vmax = get_repr_grid(jepa, normalizer)

    n_eig = min(n_eig, jepa.repr_dim)
    N = int(np.floor(np.sqrt(n_eig)))
    M = int(np.ceil(n_eig / N))

    fig, axes = plt.subplots(nrows=N, ncols=M, dpi=400, figsize=(5, 5))

    for i, ax in enumerate(axes.flat):
        # here we just plot the learned eigenvalues.
        if i >= n_eig:
            ax.axis("off")
        else:
            im = plot_dim(ax, i, m, vmin=vmin, vmax=vmax)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if not notebook:
        Logger.run().log_figure(fig, "eigvals")
        plt.close(fig)

    return fig


def log_eig_planning_plots(
    result: MPCReport,
    jepa: JEPA,
    normalizer: Normalizer,
    eigfunc_idx: int,
    idxs: List[int] = default_plot_idxs,
    close: bool = True,
    prefix: str = "",
):
    m, vmin, vmax = get_repr_grid(jepa, normalizer)
    r = get_dim_grid(m, eigfunc_idx)
    r = torch.nn.functional.pad(r, (3, 3, 3, 3), mode="constant", value=0)

    if idxs is None:
        idxs = default_plot_idxs
    for idx in idxs:
        fig = plt.figure(dpi=300)
        for i in range(result.locations.shape[0]):
            plt.subplot(6, 6, i + 1)
            plt.imshow(
                -1 * result.observations[i, idx].sum(dim=0).detach().cpu(), cmap="gray"
            )
            plt.imshow(r.detach().cpu(), alpha=0.5, vmin=vmin, vmax=vmax)
            current_location = result.locations[i, idx].detach().cpu()
            if i != result.locations.shape[0] - 1:
                # skip last one as there's no action at the last timestep
                action = result.actions[i][idx, 0].detach() * 5
                plt.arrow(
                    x=current_location[0],
                    y=current_location[1],
                    dx=action[0].cpu(),
                    dy=action[1].cpu(),
                    width=0.05,
                    color="#F77F00",
                    head_width=2,
                )
                plt.plot(
                    result.planned_locations[i][:, idx, 0, 0].detach().cpu(),
                    result.planned_locations[i][:, idx, 0, 1].detach().cpu(),
                    marker="o",
                    markersize=0.25,
                    linewidth=0.1,
                    c="red",
                    alpha=1,
                )
            # plt.scatter(
            #     start_location[0],
            #     start_location[1],
            #     s=30,
            #     c="#3777FF",
            #     marker="x",
            #     alpha=0.5,
            # )
            # plt.scatter(
            #     result.targets[idx, 0].cpu(),
            #     result.targets[idx, 1].cpu(),
            #     s=30,
            #     c="#D62828",
            #     marker="x",
            #     alpha=0.5,
            # )
            plt.xlim(0, 27)
            plt.ylim(27, 0)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

        Logger.run().log_figure(fig, f"mpc/{prefix}planning_{idx}")
        if close:
            plt.close(fig)


default_idx = [0]


def log_eigfunc_data(
    jepa: JEPA,
    normalizer: Normalizer,
    sample: WallEigenfunctionsSample,
    close: bool = True,
    idxs: List[int] = default_idx,
):
    repr_grid, vmin, vmax = get_repr_grid(jepa, normalizer)

    T = sample.states.shape[0]
    n_steps_per_option = T // sample.eigf_coeffs.shape[0]

    for k in idxs:
        fig = plt.figure(dpi=200)
        for i in range(T):
            plt.subplot(5, 4, i + 1)
            plt.imshow(sample.states[i, k].sum(dim=0).detach().cpu())
            eigf_i = i // n_steps_per_option

            if eigf_i < sample.eigf_coeffs.shape[0]:
                eigf_coeffs = sample.eigf_coeffs[eigf_i, k]
                plt.title(
                    f"{list(map(lambda x : round(x, 2), eigf_coeffs.tolist()))}",
                    fontsize=4,
                )

                background = get_product_grid(repr_grid, eigf_coeffs)
                background = torch.nn.functional.pad(
                    background, (3, 3, 3, 3), mode="constant", value=0
                )

                plt.imshow(background.detach().cpu(), alpha=0.7)

            plt.axis("off")
            if i != T - 1:
                current_location = sample.locations[i, k].cpu()
                # skip last one as there's no action at the last timestep
                action = sample.actions[i, k].detach().cpu() * 5
                plt.arrow(
                    x=current_location[0],
                    y=current_location[1],
                    dx=action[0].cpu(),
                    dy=action[1].cpu(),
                    width=0.05,
                    color="#F77F00",
                    head_width=2,
                )

        plt.tight_layout()
        Logger.run().log_figure(fig, f"sample_{k}")
        if close:
            plt.close(fig)


def plot_d4rl_eigfuncs_cont(
    jepa: JEPA,
    normalizer: Normalizer,
    notebook: bool = False,
    n_eig: int = 20,
):
    N = 100
    K = 10
    x = torch.linspace(0, K, N)
    y = torch.linspace(0, K, N)
    X, Y = torch.meshgrid(x, y, indexing="xy")

    reprs = torch.stack([X, Y], dim=-1)
    reprs_padded = torch.nn.functional.pad(reprs, (0, jepa.config.input_dim - 2))
    reprs_flat = reprs_padded.flatten(0, 1)

    reprs_flat = normalizer.normalize_state(reprs_flat)

    result = jepa.backbone(reprs_flat.cuda()).view(*reprs_padded.shape[:-1], -1)

    vmin = result.min()
    vmax = result.max()
    fig = plt.figure(dpi=200)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(result[:, :, i].detach().cpu(), vmin=vmin, vmax=vmax)
        plt.title(f"eigf_{i}", fontsize=6)
        plt.axis("off")
    plt.tight_layout()

    Logger.run().log_figure(fig, "d4rl_eigfuncs")
    if not notebook:
        plt.close(fig)

    return fig


@torch.no_grad()
def plot_d4rl_eigfuncs_samples(
    jepa: JEPA,
    dl: torch.utils.data.DataLoader,
    normalizer: Normalizer,
    notebook: bool = False,
    n_eig: int = 20,
    quick_debug: bool = False,
):
    encodings = []
    locations = []
    for batch in tqdm(dl, desc="Plotting eigfuncs samples"):
        s = batch.states.cuda().transpose(0, 1).flatten(0, 1)
        encs = jepa.backbone(s)
        encodings.append(encs.cpu())
        locations.append(batch.locations.transpose(0, 1).flatten(0, 1).cpu())

        if quick_debug and len(encodings) > 1:
            break

    encodings_t = torch.cat(encodings, dim=0)
    locations_t = torch.cat(locations, dim=0)[:, 0, :2]

    vmin = encodings_t.min()
    vmax = encodings_t.max()

    print(vmin, vmax)

    fig = plt.figure(dpi=200)
    for EIGF in range(10):
        plt.subplot(2, 5, EIGF + 1)
        indices = torch.randperm(encodings_t.shape[0])[:10000]
        vals = encodings_t[indices, EIGF].cpu()
        coords = locations_t[indices].cpu()

        plt.title(f"eigf {EIGF}", fontsize=6)
        plt.scatter(coords[:, 0], coords[:, 1], c=vals, s=3, vmin=vmin, vmax=vmax)
        plt.grid()
        plt.axis("off")
        plt.axis("square")
    plt.tight_layout()

    Logger.run().log_figure(fig, "d4rl_eigfuncs")
    if not notebook:
        plt.close(fig)

    return fig
