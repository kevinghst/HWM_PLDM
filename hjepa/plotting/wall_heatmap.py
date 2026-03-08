from typing import Mapping, Tuple

import torch
from matplotlib import pyplot as plt

from hjepa.envs.wall import DotWall
from environments.utils.normalizer import Normalizer
from hjepa.models.jepa import JEPA
from hjepa.logger import Logger

IMG_SIZE = 28
INIT_POS = (10, 3)


def plot_similarity(
    reference_enc: torch.Tensor, m: Mapping[Tuple[int, int], torch.Tensor]
):
    r = torch.zeros(21, 21).cuda()
    for i in range(3, 24):
        for j in range(3, 24):
            r[j - 3, i - 3] = (reference_enc[0] - m[(i, j)][0]).pow(2).mean()
    plt.imshow(r.detach().cpu().add(1e-1).log())
    return r


def predict_k_steps(
    jepa: JEPA,
    normalizer: Normalizer,
    start_enc: torch.Tensor,
    action: torch.Tensor,
    steps: int,
):
    action = normalizer.normalize_action(action).float().cuda()
    enc = start_enc
    for _i in range(steps):
        # assumes predictor doesn't take a latent variable.
        enc, _ = jepa.predictor(action.unsqueeze(0), enc)
    return enc


def render_pos(pos, action, K):
    env = DotWall()
    env.reset()
    pos = torch.tensor(pos).float().cuda()
    env.dot_position = pos
    env.wall_x = torch.tensor(14).cuda()
    env.hole_y = torch.tensor(14).cuda()
    env.wall_img = env._render_walls(env.wall_x, env.hole_y)

    if K > 0:
        for _i in range(K):
            obs = env.step(action.cuda())
        plt.imshow(obs.cpu().sum(dim=0), vmax=0.1)
        plt.axis("off")
    else:
        pos = torch.clamp(pos, min=2.6, max=27 - 2.6)

        dot_img = env._render_dot(pos.float().cuda())
        plt.imshow(dot_img.cpu() + env.wall_img.cpu() * dot_img.max().cpu(), vmax=0.1)
        plt.axis("off")


def plot_wall_heatmap(
    jepa: JEPA,
    prober_model: torch.nn.Module,
    normalizer: Normalizer,
    steps: int = 16,
    notebook: bool = False,
    n_eig: int = 20,
):
    env = DotWall()
    current_state = env.reset()

    current_state_normalized = normalizer.normalize_state(current_state.unsqueeze(0))

    wall_img = env._render_walls(torch.tensor(14).cuda(), torch.tensor(14).cuda())
    dot_img = env._render_dot(torch.tensor(INIT_POS).float().cuda())
    # plt.imshow(dot_img.cpu() + wall_img.cpu() * dot_img.max().cpu())

    # build a map of representations at all locations for nearest neighbor heatmap
    m = {}
    for i in range(3, 24):
        for j in range(3, 24):
            dot_position = torch.tensor([i, j]).float().cuda()
            dot_img = env._render_dot(dot_position)
            obs = torch.stack([dot_img, wall_img * dot_img.max()], dim=0)
            current_state_normalized = normalizer.normalize_state(obs.unsqueeze(0))
            enc = jepa.backbone(current_state_normalized.cuda())
            m[(i, j)] = enc

    es = m[INIT_POS]

    fsize = 4
    es = m[INIT_POS]
    fig = plt.figure(dpi=400, figsize=(5, 2.5))
    plt.subplot(2, 6, 1)
    plot_similarity(es, m)
    plt.title("start", fontsize=fsize)
    plt.axis("off")

    pos = normalizer.unnormalize_location(prober_model(es)).detach().cpu()

    plt.subplot(2, 6, 7)
    render_pos(INIT_POS, None, 0)
    plt.scatter(pos[0, 0, 0], pos[0, 0, 1], marker="x", color="red", alpha=0.8, s=10)

    actions = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]])

    K = 16
    for i in range(len(actions)):
        plt.subplot(2, 6, 2 + i)
        representation = predict_k_steps(jepa, normalizer, es, actions[i], steps=K)
        plot_similarity(representation, m)
        pos = (
            normalizer.unnormalize_location(prober_model(representation)).detach().cpu()
        )
        plt.title(str(actions[i].tolist()), fontsize=fsize)
        plt.axis("off")
        plt.subplot(2, 6, 8 + i)
        render_pos(INIT_POS, actions[i], K)
        plt.scatter(
            pos[0, 0, 0], pos[0, 0, 1], marker="x", color="red", alpha=0.8, s=10
        )

    plt.subplot(2, 6, 6)

    check = (23, 3)
    es2 = m[check]
    plot_similarity(es2, m)
    plt.axis("off")

    plt.subplot(2, 6, 12)
    render_pos(check, None, 0)

    plt.tight_layout()

    if not notebook:
        Logger.run().log_figure(fig, f"heatmap_{steps}")
        plt.close(fig)

    return fig
