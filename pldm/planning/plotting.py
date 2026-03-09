from pldm.logger import Logger
from typing import List
from matplotlib import pyplot as plt
from pldm.logger import Logger
import math
from tqdm import tqdm
import numpy as np
import torch
import sys

default_plot_idxs = list(range(100))


def log_planning_plots(
    result,
    report,
    idxs: List[int] = default_plot_idxs,
    prefix: str = "wall_",
    n_steps: int = 44,
    xy_action: bool = True,
    plot_every: int = 1,
    quick_debug: bool = False,
    pixel_mapper=None,
    plot_failure_only: bool = False,
    log_pred_dist_every: int = sys.maxsize,
    mark_action: bool = True,
    plot_l2_predictions: bool = True,
):
    num_plots = math.ceil(len(result.locations) / plot_every)
    grid_size = max(4, math.ceil(math.sqrt(num_plots)))

    img_size = result.observations[0][0].shape[-1]

    if pixel_mapper is not None:
        targets_pixels = torch.as_tensor(
            pixel_mapper.obs_coord_to_pixel_coord(result.targets)
        )

        locations_pixels = [
            torch.as_tensor(pixel_mapper.obs_coord_to_pixel_coord(x))
            for x in result.locations
        ]

        pred_locations_pixels = [
            torch.as_tensor(pixel_mapper.obs_coord_to_pixel_coord(x))
            for x in result.pred_locations
        ]

        # Handle L2 predictions for hierarchical planning
        if (
            plot_l2_predictions
            and hasattr(result, "pred_locations_l2")
            and result.pred_locations_l2
        ):
            pred_locations_l2_pixels = [
                torch.as_tensor(pixel_mapper.obs_coord_to_pixel_coord(x))
                for x in result.pred_locations_l2
            ]
        else:
            pred_locations_l2_pixels = None
    else:
        targets_pixels = result.targets
        locations_pixels = result.locations
        pred_locations_pixels = result.pred_locations

        # Handle L2 predictions for hierarchical planning
        if (
            plot_l2_predictions
            and hasattr(result, "pred_locations_l2")
            and result.pred_locations_l2
        ):
            pred_locations_l2_pixels = result.pred_locations_l2
        else:
            pred_locations_l2_pixels = None

    if idxs is None:
        idxs = default_plot_idxs
    for idx in idxs:
        if plot_failure_only and report.success[idx]:
            continue

        fig = plt.figure(dpi=300)
        start_location = locations_pixels[0][idx].cpu()
        subplot_idx = 0
        for i in range(len(locations_pixels)):
            if i % plot_every:
                continue

            if i > report.terminations[idx]:
                break
            plt.subplot(grid_size, grid_size, subplot_idx + 1)

            if "wall" in prefix:
                img = -1 * result.observations[i][idx].sum(dim=0).detach().cpu()
            elif result.observations[i][idx].shape[0] > 1:
                # if multiple channels, need to convert to grayscale
                img = result.observations[i][idx].detach().cpu().numpy()  # (3, 64, 64)
                img = np.transpose(img, (1, 2, 0))
                # Convert to grayscale using the weighted sum of RGB channels
                img = (
                    0.2989 * img[:, :, 0]
                    + 0.5870 * img[:, :, 1]
                    + 0.1140 * img[:, :, 2]
                )

                # Normalize the grayscale image to the range [0, 1]
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = result.observations[i][idx][0]

            plt.imshow(
                img,
                cmap="gray",
            )
            current_location = locations_pixels[i][idx].detach().cpu()
            if i != len(locations_pixels) - 1:
                # skip last one as there's no action at the last timestep
                action = result.action_history[i][idx, 0].detach()

                action = action * 5  # for visibility

                if mark_action:
                    plt.arrow(
                        x=current_location[0],
                        y=current_location[1],
                        dx=action[0].cpu(),
                        dy=action[1].cpu(),
                        width=0.05,
                        color="#F77F00",
                        head_width=2,
                    )

                if pred_locations_pixels is not None:
                    plt.plot(
                        pred_locations_pixels[i][:, idx, 0].detach().cpu(),
                        pred_locations_pixels[i][:, idx, 1].detach().cpu(),
                        marker="o",
                        markersize=0.1,
                        linewidth=0.1,
                        c="red",
                        alpha=1,
                    )

                    if log_pred_dist_every < 999999:
                        final_pred_dists = result.final_preds_dist[i][:, idx].tolist()
                        # skip every nth. make sure to include first and last
                        last_dist = final_pred_dists[-1]
                        final_pred_dists = final_pred_dists[::log_pred_dist_every]
                        if last_dist != final_pred_dists[-1]:
                            final_pred_dists.append(last_dist)

                        plt.text(
                            1.05,
                            0.5,
                            "\n".join([f"{dist:.2f}" for dist in final_pred_dists]),
                            transform=plt.gca().transAxes,
                            fontsize=2.5,
                            verticalalignment="center",
                            horizontalalignment="left",
                            color="blue",
                        )

                # Plot L2 predictions
                if pred_locations_l2_pixels is not None and i < len(
                    pred_locations_l2_pixels
                ):
                    plt.plot(
                        pred_locations_l2_pixels[i][:, idx, 0].detach().cpu(),
                        pred_locations_l2_pixels[i][:, idx, 1].detach().cpu(),
                        marker="x",
                        markersize=1,
                        linewidth=0.5,
                        c="green",
                        alpha=1,
                    )

            plt.scatter(
                start_location[0],
                start_location[1],
                s=0.1,
                c="blue",
                marker="o",
                alpha=1,
            )

            plt.scatter(
                targets_pixels[idx, 0].cpu(),
                targets_pixels[idx, 1].cpu(),
                s=0.1,
                c="#F77F00",
                marker="o",
                alpha=1,
            )
            plt.xlim(0, img_size - 1)
            plt.ylim(img_size - 1, 0)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            subplot_idx += 1

        # if quick_debug:
        #     breakpoint()
        # result.targets.shape = [15, 2]
        # result.locations[0].shape = [15, 2]

        log_name = f"mpc/{prefix}_{idx}"
        if plot_failure_only:
            start_x = result.locations[0][idx][0]
            start_y = result.locations[0][idx][1]
            target_x = result.targets[idx][0]
            target_y = result.targets[idx][1]
            log_name += (
                f"_{int(start_x)}_{int(start_y)}_{int(target_x)}_{int(target_y)}"
            )

        Logger.run().log_figure(fig, log_name)
        plt.close(fig)


def log_l2_planning_loss(
    planning_result,
    prefix: str = "",
):
    if planning_result.loss_history_l2[0] is None:
        return

    logger = Logger.run()
    steps = [0, len(planning_result.loss_history_l2) // 2]

    for step in steps:
        losses = planning_result.loss_history_l2[step]
        for loss in losses:
            log_dict = {f"{prefix}_l2_step_{step}_plan_loss": loss}
            logger.log(log_dict)
        logger.log({f"{prefix}_l2_step_{step}_plan_iterations": len(losses)})

    for step in steps:
        losses = planning_result.loss_history[step]
        for loss in losses:
            log_dict = {f"{prefix}_l2_l1_step_{step}_plan_loss": loss}
            logger.log(log_dict)
        logger.log({f"{prefix}_l2_l1_step_{step}_plan_iterations": len(losses)})


def log_hierarchical_planning_plots(
    result,
    report,
    idxs: List[int] = default_plot_idxs,
    img_size: int = 28,
    prefix: str = "",
    n_steps: int = 44,
    border_wall_loc: int = 5,
    remove_padding: bool = False,
    xy_action: bool = True,
):
    grid_size = max(4, math.ceil(math.sqrt(len(result.locations))))

    if idxs is None:
        idxs = default_plot_idxs
    for j in tqdm(idxs, desc="Plotting"):
        # for each sample
        fig = plt.figure(figsize=(10, 10), dpi=300)
        start_location = result.locations[0][j].cpu()
        for i in range(len(result.locations)):
            # for each action step
            if i > report.terminations[j]:
                break
            plt.subplot(grid_size, grid_size, i + 1)
            obs = result.observations[i][j]
            if remove_padding:
                obs = obs[
                    :,
                    border_wall_loc - 1 : img_size - border_wall_loc + 1,
                    border_wall_loc - 1 : img_size - border_wall_loc + 1,
                ]
            plt.imshow(-1 * obs.sum(dim=0).detach().cpu(), cmap="gray")
            plt.scatter(
                start_location[0],
                start_location[1],
                s=30,
                c="#3777FF",
                marker="x",
                alpha=0.5,
            )
            plt.scatter(
                result.targets[j, 0].cpu(),
                result.targets[j, 1].cpu(),
                s=30,
                c="#D62828",
                marker="x",
                alpha=0.5,
            )

            if i != len(result.locations) - 1:
                current_location = result.locations[i][j].detach().cpu()
                action = result.action_history[i][j, 0].detach()

                action = action * 5  # for visibility

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
                    result.pred_locations[i][:, j, 0].cpu(),
                    result.pred_locations[i][:, j, 1].cpu(),
                    marker="o",
                    markersize=0.25,
                    linewidth=0.1,
                    c="red",
                    alpha=1,
                )

                if i < len(result.pred_locations_l2):
                    # l2_depth = result.pred_depths_history_l2[i][j]
                    l2_final_pred_dists = result.final_preds_dist[i][:, j].tolist()
                    last_dist = l2_final_pred_dists[-1]
                    skip_n = 2
                    l2_final_pred_dists = l2_final_pred_dists[::skip_n]
                    if last_dist != l2_final_pred_dists[-1]:
                        l2_final_pred_dists.append(last_dist)

                    # l2 planning could have terminated early
                    plt.plot(
                        result.pred_locations_l2[i][:, j, 0].cpu(),
                        result.pred_locations_l2[i][:, j, 1].cpu(),
                        marker="x",
                        markersize=1,
                        linewidth=0.5,
                        c="green",
                        alpha=1,
                    )

                    plt.text(
                        1.05,
                        0.5,
                        "\n".join([f"{dist:.2f}" for dist in l2_final_pred_dists]),
                        transform=plt.gca().transAxes,
                        fontsize=2.5,
                        verticalalignment="center",
                        horizontalalignment="left",
                        color="blue",
                    )

            if remove_padding:
                plt.xlim(0, img_size - border_wall_loc * 2 + 1)
                plt.ylim(img_size - border_wall_loc * 2 + 1, 0)
            else:
                plt.xlim(0, img_size - 1)
                plt.ylim(img_size - 1, 0)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

        plt.tight_layout()

        if report.unsucc_trials_with_succ_plans[j]:
            Logger.run().log_figure(fig, f"hmpc/{prefix}planning_{j}_X")
        else:
            Logger.run().log_figure(fig, f"hmpc/{prefix}planning_{j}")

        plt.close(fig)

def log_l1_planning_loss(result, prefix: str = "wall_"):
    logger = Logger.run()
    steps = [0, len(result.loss_history) // 2]

    for step in steps:
        losses = result.loss_history[step]
        for loss in losses:
            log_dict = {f"{prefix}_l1_step_{step}_plan_loss": loss}
            logger.log(log_dict)
        logger.log({f"{prefix}_l1_step_{step}_plan_iterations": len(losses)})
