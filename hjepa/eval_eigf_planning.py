import dataclasses
from dataclasses import dataclass
from typing import Optional, List, NamedTuple
from pathlib import Path
import random

from omegaconf import MISSING
import torch
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt

from hjepa.configs import ConfigBase
from hjepa.logger import Logger
from planning.mpc import PlanningConfig, plan_level1
from planning.objectives import EigfObjective, BaseMPCObjective
import utils
from hjepa.envs.wall import DotWall
from plotting.eigfuncs import get_repr_grid, get_dim_grid
from environments.utils.normalizer import Normalizer


@dataclass
class EvalEigfPlanningConfig(ConfigBase):
    checkpoint_path: str = MISSING
    output_path: Optional[str] = None
    quick_debug: bool = False
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    wandb: bool = False
    log_plots: bool = True
    seed: int = 42
    eval_batches_consistency: int = 30
    consistency_noise_std: float = 0.2
    eval_batches_value: int = 5
    planning: PlanningConfig = PlanningConfig()


def get_obj_vals(t, observations, objective):
    normed_obs = t.ds.normalizer.normalize_state(observations)
    # unsqueeze for time
    encoding = t.model.level1.backbone(normed_obs).unsqueeze(0)
    vals = objective(encoding, sum_batch=False)
    return vals


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class UnfoldPlanningResult(NamedTuple):
    actions: List[torch.Tensor]
    observations: List[torch.Tensor]
    locations: List[torch.Tensor]


def batched_unfold_planning(
    jepa: torch.nn.Module,
    prober: torch.nn.Module,
    n_steps: int,
    normalizer: Normalizer,
    planning_config: PlanningConfig,
    envs: List[DotWall],
    init_obs: torch.Tensor,
    objective: BaseMPCObjective,
):
    obs = init_obs
    action_history = []
    observation_history = [obs]
    location_history = [torch.stack([env.dot_position for env in envs])]

    for i in range(n_steps):
        actions, planned_locations, losses = plan_level1(
            jepa=jepa,
            normalizer=normalizer,
            prober=prober,
            current_state=obs,
            objective=objective,
            n_iters=planning_config.n_iters,
            lr=planning_config.lr,
            max_step=planning_config.max_step,
            plan_length=min(
                planning_config.max_plan_length, planning_config.n_steps - i
            ),
            l2_reg=planning_config.l2_reg,
            action_change_reg=planning_config.action_change_reg,
        )
        obs = torch.stack([env.step(action[0]) for env, action in zip(envs, actions)])
        action_history.append(actions.detach())
        observation_history.append(obs.detach())
        location_history.append(
            torch.stack([env.dot_position for env in envs]).detach()
        )

    return UnfoldPlanningResult(
        actions=action_history,
        observations=observation_history,
        locations=location_history,
    )


def eval_performance(t, prober, config, closefig=True):
    repr_grid, vmin, vmax = get_repr_grid(t.model.level1, t.ds.normalizer)

    # measuring performance
    all_diffs = []

    rng = np.random.default_rng(42)

    num_plots = 8
    batch_plots = 10

    for batch_idx in tqdm(range(config.eval_batches_value)):
        option_length = 4  # Number of steps we follow the given eigfunction for.

        pcfg_c = dataclasses.replace(config.planning, n_envs=1024, n_steps=4)

        envs = [
            DotWall(rng=rng, wall_config=t.config.wall_config)
            for _ in range(pcfg_c.n_envs)
        ]
        obs = torch.stack([env.reset().cuda() for env in envs])

        eigf_idxs = torch.randint(low=1, high=4, size=(len(envs),)).cuda()
        minimize = (torch.rand(len(envs)) > 0.5).cuda()
        objective = EigfObjective(
            eigf_idxs, minimize=minimize, discount=pcfg_c.discount, sum_all_diffs=True
        )

        unfold_result = batched_unfold_planning(
            jepa=t.model.level1,
            prober=prober,
            n_steps=option_length,
            normalizer=t.ds.normalizer,
            planning_config=pcfg_c,
            envs=envs,
            init_obs=obs,
            objective=objective,
        )
        all_diffs.append(
            get_obj_vals(t, unfold_result.observations[-1], objective)
            - get_obj_vals(t, unfold_result.observations[0], objective)
        )

        if batch_idx < batch_plots:
            fig = plt.figure(dpi=400)

            all_locations_t = torch.stack(unfold_result.locations).cpu()
            arrows = all_locations_t[-1, :] - all_locations_t[0, :]
            for j in range(num_plots):
                plt.subplot(2, 4, j + 1)
                plt.quiver(
                    all_locations_t[0, j, 0].numpy(),
                    all_locations_t[0, j, 1].numpy(),
                    arrows[j, 0].numpy(),
                    arrows[j, 1].numpy(),
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.01,
                )
                plt.scatter(
                    all_locations_t[-1, j, 0].numpy(),
                    all_locations_t[-1, j, 1].numpy(),
                    s=0.1,
                    c="r",
                )
                plt.scatter(
                    all_locations_t[0, j, 0].numpy(),
                    all_locations_t[0, j, 1].numpy(),
                    s=0.1,
                    c="magenta",
                )

                background = get_dim_grid(repr_grid, eigf_idxs[j])
                background = torch.nn.functional.pad(
                    background, (3, 3, 3, 3), mode="constant", value=0
                )

                plt.imshow(background.detach().cpu(), alpha=0.7)
                plt.axis("off")
                plt.title(
                    f"{eigf_idxs[batch_idx]} min={minimize[batch_idx]}", fontsize=4
                )

            plt.tight_layout()
            Logger.run().log_figure(fig, f"perf_{batch_idx}")

            if closefig:
                plt.close(fig)

    return torch.cat(all_diffs).mean()


def eval_consistency(t, prober, config: EvalEigfPlanningConfig, closefig: bool = True):
    repr_grid, vmin, vmax = get_repr_grid(t.model.level1, t.ds.normalizer)

    rng = np.random.default_rng(42)

    eigf_idxs = rng.integers(low=1, high=4, size=(config.eval_batches_consistency,))
    minimize = rng.choice(2, config.eval_batches_consistency).astype(bool)
    positions = rng.uniform(
        low=2.6, high=27 - 2.6, size=(config.eval_batches_consistency, 2)
    ).astype(np.float32)

    final_locations_stds = []

    fig = plt.figure(dpi=400)
    num_plots = 8

    for batch_idx in range(config.eval_batches_consistency):
        option_length = 4  # Number of steps we follow the given eigfunction for.

        if batch_idx < num_plots:
            plt.subplot(2, 4, batch_idx + 1)

        pcfg_c = dataclasses.replace(config.planning, n_envs=32, n_steps=4)

        envs = [
            DotWall(rng=rng, wall_config=t.config.wall_config)
            for _ in range(pcfg_c.n_envs)
        ]
        obs = torch.stack(
            [
                env.reset(
                    (
                        torch.tensor(positions[batch_idx])
                        + torch.randn(2) * config.consistency_noise_std
                    ).cuda()
                )
                for env in envs
            ]
        )

        objective = EigfObjective(
            torch.tensor([eigf_idxs[batch_idx]]).cuda(),
            minimize=torch.tensor([minimize[batch_idx]]).cuda(),
        )

        unfold_result = batched_unfold_planning(
            jepa=t.model.level1,
            prober=prober,
            n_steps=option_length,
            normalizer=t.ds.normalizer,
            planning_config=pcfg_c,
            envs=envs,
            init_obs=obs,
            objective=objective,
        )

        final_locations_stds.append(unfold_result.locations[-1].std(dim=0).mean())

        if batch_idx < num_plots:
            all_locations_t = torch.stack(unfold_result.locations).cpu()
            arrows = all_locations_t[-1, :] - all_locations_t[0, :]
            plt.quiver(
                all_locations_t[0, :, 0].numpy(),
                all_locations_t[0, :, 1].numpy(),
                arrows[:, 0].numpy(),
                arrows[:, 1].numpy(),
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.01,
                alpha=0.1,
            )
            plt.scatter(
                all_locations_t[-1, :, 0].numpy(),
                all_locations_t[-1, :, 1].numpy(),
                s=0.1,
                c="r",
                alpha=0.5,
            )
            plt.scatter(
                all_locations_t[0, :, 0].numpy(),
                all_locations_t[0, :, 1].numpy(),
                s=0.1,
                c="magenta",
                alpha=0.5,
            )
            plt.scatter(
                positions[batch_idx][0], positions[batch_idx][1], s=0.5, c="blue"
            )

            background = get_dim_grid(repr_grid, eigf_idxs[batch_idx])
            background = torch.nn.functional.pad(
                background, (3, 3, 3, 3), mode="constant", value=0
            )

            plt.imshow(background.detach().cpu(), alpha=0.7)
            plt.axis("off")
            plt.title(f"{eigf_idxs[batch_idx]} min={minimize[batch_idx]}", fontsize=4)

    plt.tight_layout()
    Logger.run().log_figure(fig, "consistency")

    if closefig:
        plt.close(fig)
        return torch.stack(final_locations_stds).mean()
    else:
        return torch.stack(final_locations_stds).mean(), fig


def main(config: EvalEigfPlanningConfig):
    Logger.run().initialize(
        output_path=config.output_path,
        wandb_enabled=config.wandb,
        project="HJEPA-eigf-planning",
        name=config.run_name,
        group=config.run_group,
        config=dataclasses.asdict(config),
    )

    torch.set_num_threads(1)
    seed_everything(config.seed)

    t = utils.load_model(config.checkpoint_path)

    prober = utils.load_prober(
        t.model.level1.config.repr_dim,
        t.config.probing_cfg.prober_arch,
        output_shape=[1, 2],  # hardcoded for now, change if more dots are added
        path=Path(config.checkpoint_path),
    )

    t.model.eval()

    t.config.probing_cfg.epochs = 100

    if config.quick_debug:
        config.planning.n_iters = 2
        config.planning.n_envs = 4
        config.eval_batches_consistency = 1
        config.eval_batches_value = 1

    perf = eval_performance(t, prober, config).item()
    print("Avg perf:", round(perf, 3))
    Logger.run().log_summary({"avg_perf": perf})

    consistency = eval_consistency(t, prober, config).item()
    print("Avg consistency:", round(consistency, 3))
    Logger.run().log_summary({"avg_consistency": consistency})


if __name__ == "__main__":
    config = EvalEigfPlanningConfig.parse_from_command_line()
    main(config)
