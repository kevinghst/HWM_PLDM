from typing import NamedTuple, Optional, List
from dataclasses import dataclass, field

from tqdm import tqdm
import torch
import time

from environments.utils.normalizer import Normalizer
from environments.wall.data.wall import WallDatasetConfig
from hjepa.configs import ConfigBase

from hjepa.models.hjepa import HJEPA
from hjepa.utils import format_seconds
from .utils import *
from hjepa.planning.utils import get_lr_p_results
from hjepa.planning.plotting import (
    log_l2_planning_loss,
    log_hierarchical_planning_plots,
)
from hjepa.planning.planners.enums import PlannerType, PlannerConfig
from hjepa.planning.utils import normalize_actions
import numpy as np
from hjepa.planning.wall.enums import HierarchicalWallMPCConfig, HierarchicalMPCReport
from hjepa.planning.wall.mpc import WallMPCEvaluator
from hjepa.planning.enums import PooledMPCResult


class HierarchicalWallMPCEvaluator(WallMPCEvaluator):
    def __init__(
        self,
        config: HierarchicalWallMPCConfig,
        model: HJEPA,
        prober: torch.nn.Module,
        prober_l2: torch.nn.Module,
        normalizer: Normalizer,
        wall_config: WallDatasetConfig,
        cross_wall: bool = True,
        quick_debug: bool = False,
        prefix: str = "wall_",
    ):
        super().__init__(
            config=config,
            model=model,
            prober=prober,
            normalizer=normalizer,
            wall_config=wall_config,
            cross_wall=cross_wall,
            quick_debug=quick_debug,
            prefix=prefix,
            hierarchical=True,
        )

        self.prober_l2 = prober_l2

    def evaluate(self):
        start_time = time.time()

        data = self._perform_mpc_in_chunks()

        elapsed_time = int(time.time() - start_time)
        print(f"hmpc planning took {format_seconds(elapsed_time)}")

        report = self._construct_report(data, elapsed_time=elapsed_time)

        log_l2_planning_loss(data, prefix=self.prefix)

        if self.config.visualize_planning:
            log_hierarchical_planning_plots(
                result=data,
                report=report,
                idxs=[0, 1] if self.quick_debug else list(range(len(report.errors))),
                img_size=self.wall_config.img_size,
                prefix=self.prefix,
                n_steps=self.config.n_steps,
                border_wall_loc=self.wall_config.border_wall_loc,
                xy_action=self.wall_config.action_param_xy,
            )

        return data, report

    def _construct_report(self, data: PooledMPCResult, elapsed_time: float = 0):
        """
        Run various analytics on mpc result
        """
        config = self.config
        wall_locs = self.wall_locs
        door_locs = self.door_locs

        locations = data.locations
        targets = data.targets

        terminations = determine_terminations(
            locations, targets, config.error_threshold
        )

        final_errors = torch.stack(
            [
                (
                    locations[min(terminations[i], len(locations) - 1)][i].cpu()
                    - target.cpu()
                )
                .pow(2)
                .mean()
                for i, target in enumerate(targets)
            ]
        )

        # percentage of trials where agent gets to the other side of the wall, given it started from a different side than the target
        locations_t = torch.stack(locations)
        starts, ends = locations_t[0], locations_t[-1]
        cross_wall_rate = calculate_cross_wall_rate(
            starts=starts,
            ends=ends,
            wall_locs=wall_locs,
            targets=targets,
            wall_config=self.wall_config,
        )

        # percentage of trials where agent's initial plan reaches other sideof the wall, given it started from different side of the target

        init_plan_last_loc = data.pred_locations_l2[0][-1]
        init_plan_cross_wall_rate = calculate_cross_wall_rate(
            starts=starts,
            ends=init_plan_last_loc,
            wall_locs=wall_locs,
            targets=targets,
            wall_config=self.wall_config,
        )

        # Calculate the followings:
        # number of plans that get model close enuf to goal
        # number of plans that get model close enuf to goal while NOT crossing walls
        analyze_first_n_steps = 20

        reach_target_plans, legal_plans = calculate_num_succ_plans(
            data.pred_locations_l2,
            wall_locs=wall_locs,
            door_locs=door_locs,
            targets=targets,
            goal_threshold=5,
            wall_config=self.wall_config,
            first_n_steps=analyze_first_n_steps,
        )

        unsucc_plans = 1 - reach_target_plans
        succ_legal_plans = legal_plans & reach_target_plans
        succ_illegal_plans = (1 - legal_plans) & reach_target_plans
        total_plans = torch.tensor(reach_target_plans.shape).prod().item()

        # calculate avg final pred repr distance to goal for various plan types
        last_preds_dist = torch.stack(
            [x[-1] for x in data.final_preds_dist[:analyze_first_n_steps]]
        )

        succ_legal_plans_avg_dist = (
            succ_legal_plans * last_preds_dist
        ).sum() / succ_legal_plans.sum()
        succ_illegal_plans_avg_dist = (
            succ_illegal_plans * last_preds_dist
        ).sum() / succ_illegal_plans.sum()
        unsucc_plans_avg_dist = (
            unsucc_plans * last_preds_dist
        ).sum() / unsucc_plans.sum()

        # calcualte avg start to goal norm and angle diffs for successful vs unsuccessful trials
        succ_trials = final_errors < 1
        unsucc_trials = ~(succ_trials)
        succ_norm_diff, succ_angle_diff = analyze_norm_angle_diff(
            starts[succ_trials],
            targets[succ_trials],
            wall_locs=wall_locs[succ_trials],
            door_locs=door_locs[succ_trials],
        )
        unsucc_norm_diff, unsucc_angle_diff = analyze_norm_angle_diff(
            starts[unsucc_trials],
            targets[unsucc_trials],
            wall_locs=wall_locs[unsucc_trials],
            door_locs=door_locs[unsucc_trials],
        )
        trials_with_succ_plans = succ_legal_plans.any(dim=0).int()
        unsucc_trials_with_succ_plans = trials_with_succ_plans.cpu() & unsucc_trials

        # hacky. figure out why its breaking later...
        try:
            norm_diff_p = get_lr_p_results(
                torch.cat([succ_norm_diff, unsucc_norm_diff]),
                torch.cat(
                    [
                        torch.ones_like(succ_norm_diff),
                        torch.zeros_like(unsucc_norm_diff),
                    ]
                ),
            )
        except:
            norm_diff_p = 0

        try:
            angle_diff_p = get_lr_p_results(
                torch.cat([succ_angle_diff, unsucc_angle_diff]),
                torch.cat(
                    [
                        torch.ones_like(succ_angle_diff),
                        torch.zeros_like(unsucc_angle_diff),
                    ]
                ),
            )
        except:
            angle_diff_p = 0

        report = HierarchicalMPCReport(
            error_mean=final_errors.mean(),
            errors=final_errors,
            terminations=terminations,
            planning_time=elapsed_time,
            cross_wall_rate=cross_wall_rate,
            init_plan_cross_wall_rate=init_plan_cross_wall_rate,
            reach_target_plans_rate=reach_target_plans.sum().item() / total_plans,
            succ_plans_rate=succ_legal_plans.sum().item() / total_plans,
            succ_illegal_plans_rate=succ_illegal_plans.sum().item() / total_plans,
            unsucc_trials_with_succ_plans=unsucc_trials_with_succ_plans,
            unsucc_trials_with_succ_plans_rate=unsucc_trials_with_succ_plans.sum()
            / unsucc_trials.sum(),
            succ_legal_plans_avg_dist=succ_legal_plans_avg_dist,
            succ_illegal_plans_avg_dist=succ_illegal_plans_avg_dist,
            unsucc_plans_avg_dist=unsucc_plans_avg_dist,
            succ_plans_avg_norm_diff=(
                succ_norm_diff.mean() if succ_norm_diff.shape[0] else 0
            ),
            succ_plans_avg_angle_diff=(
                succ_angle_diff.mean() if succ_angle_diff.shape[0] else 0
            ),
            unsucc_plans_avg_norm_diff=(
                unsucc_norm_diff.mean() if unsucc_norm_diff.shape[0] else 0
            ),
            unsucc_plans_avg_angle_diff=(
                unsucc_angle_diff.mean() if unsucc_angle_diff.shape[0] else 0
            ),
            norm_diff_p=norm_diff_p,
            angle_diff_p=angle_diff_p,
        )

        return report

    def determine_optimal_depths(
        self,
        obs,
        targets,
        target_repr,
        planner,
    ):
        config = self.config.level2
        hjepa = self.model
        # probe_depth_pred_locations = []
        batch_size = obs.shape[0]
        print("Determining optimal l2 prediction depth...")
        dist_to_target_repr = torch.full((batch_size, config.max_plan_length + 1), 1e9)
        for depth in range(config.min_plan_length, config.max_plan_length + 1):
            # get representation of observation
            enc1 = hjepa.level1.backbone(obs.cuda()).encodings
            enc2 = hjepa.level2.backbone(enc1).encodings.detach()

            if config.planner_type == PlannerType.SGD:
                preds_l2, _, _, _ = planner.plan(
                    current_state=enc2,
                    steps_left=depth,
                    repr_input=True,
                )
            else:
                preds_l2, _, _, _ = planner.plan(
                    current_state=enc2,
                    plan_size=depth,
                )

            l2_dist = torch.norm(target_repr - preds_l2[-1], dim=1)
            dist_to_target_repr[:, depth] = l2_dist.detach()
            # probe_depth_pred_locations.append(locations_l2[-1])

        close_enough = (dist_to_target_repr < config.depth_probe_threshold).type(
            torch.int
        )
        opt_depths = torch.argmin(dist_to_target_repr, dim=1)  # (bs,)
        opt_depths[torch.any(close_enough, dim=1)] = torch.argmax(close_enough, dim=1)[
            torch.any(close_enough, dim=1)
        ]
        opt_steps = opt_depths * hjepa.config.step_skip

        # analyze pred embed dists
        # probe_depth_pred_locations = torch.stack(probe_depth_pred_locations)

        dist_to_target_repr = dist_to_target_repr.transpose(0, 1)[1:]
        # dist_to_target = torch.norm(
        #     probe_depth_pred_locations - targets.unsqueeze(0), dim=-1
        # )
        dist_to_target_repr = dist_to_target_repr.reshape(-1)
        # dist_to_target = dist_to_target.reshape(-1)

        # ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7)]
        # for rg in ranges:
        #     a, b = rg
        #     mask = (dist_to_target >= a) & (dist_to_target <= b)
        #     mask = mask.to(dist_to_target_repr.device)
        #     count = mask.sum().item()
        #     selected_target_repr = dist_to_target_repr[mask]
        # if count:
        #     print(rg)
        #     print(count)
        #     print(torch.mean(selected_target_repr))
        #     print(torch.max(selected_target_repr))
        #     print(torch.min(selected_target_repr))

        return opt_steps
