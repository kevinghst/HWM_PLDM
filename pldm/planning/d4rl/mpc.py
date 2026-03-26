from typing import Optional

import torch

from pldm.models.hjepa import HJEPA

from pldm_envs.utils.normalizer import Normalizer
from pldm.planning.utils import *
from pldm.planning.plotting import log_planning_plots, log_l1_planning_loss
from pldm.planning.d4rl.enums import (
    D4RLMPCConfig,
    MPCReport,
)
from pldm.planning.mpc import MPCEvaluator
from pldm.planning.enums import MPCResult, PooledMPCResult

from pldm_envs.diverse_maze.utils import *
from pldm_envs.diverse_maze.utils import PixelMapper
from pldm_envs.diverse_maze.evaluation.maze2d_envs_generator import (
    Maze2DEnvsGenerator,
)
from pldm_envs.diverse_maze.wrappers import *


class MazeMPCEvaluator(MPCEvaluator):
    def __init__(
        self,
        config: D4RLMPCConfig,
        normalizer: Normalizer,
        model: HJEPA,
        pixel_mapper: PixelMapper,
        prober: Optional[torch.nn.Module] = None,
        prober_l2: Optional[torch.nn.Module] = None,
        fix_start: bool = False,
        prefix: str = "d4rl_",
        quick_debug: bool = False,
        l2_use_latent_mean_std: bool = False,
    ):
        super().__init__(
            config=config,
            model=model,
            prober=prober,
            normalizer=normalizer,
            quick_debug=quick_debug,
            prefix=prefix,
            pixel_mapper=pixel_mapper,
            l2_use_latent_mean_std=l2_use_latent_mean_std,
        )

        level_cfg = getattr(config, config.level)

        envs_generator = Maze2DEnvsGenerator(
            env_name=config.env_name,
            n_envs=level_cfg.n_envs,
            min_block_radius=level_cfg.min_block_radius,
            max_block_radius=level_cfg.max_block_radius,
            action_repeat=config.action_repeat,
            action_repeat_mode=config.action_repeat_mode,
            seed=config.seed,
            stack_states=config.stack_states,
            image_obs=config.image_obs,
            data_path=config.data_path,
            trials_path=config.set_start_target_path,
            unique_shortest_path=config.unique_shortest_path,
            normalizer=self.normalizer,
        )
        self.envs, _ = envs_generator()

        self.prober_l2 = prober_l2
        self.fix_start = fix_start

    def close(self):
        print(f"closing {self.prefix}")
        """Manually close environments."""
        for env in self.envs:
            if env is not None:
                env.close()

    def _get_opt_path_sim(self, data: MPCResult):
        raise NotImplementedError

        for i in range(len(self.envs)):
            env = self.envs[i]

            if not contains_wrapper(env, NavigationWrapper):
                env = NavigationWrapper(env)

    def _construct_ood_report(self, data: PooledMPCResult, successes):
        ood_report = {}

        for i, env in enumerate(self.envs):
            key = f"{env.mode}_{env.ood_dist}"
            if key in ood_report:
                ood_report[key].append(successes[i])
            else:
                ood_report[key] = [successes[i]]

        for key in ood_report:
            ood_report[key] = sum(ood_report[key]) / len(ood_report[key])

        return ood_report

    def _construct_report(self, data: PooledMPCResult):
        # Determine termination indices
        T = len(data.reward_history)
        B = data.reward_history[0].shape[0]

        terminations = [T] * B

        for b_i in range(B):
            for t_i in range(T):
                if data.reward_history[t_i][b_i]:
                    terminations[b_i] = t_i
                    break

        successes = [int(x < T) for x in terminations]
        success_rate = sum(successes) / len(successes)

        num_turns = [x.turns for x in self.envs]

        one_turn_successes = [
            successes[i] for i in range(len(num_turns)) if num_turns[i] == 1
        ]
        two_turn_successes = [
            successes[i] for i in range(len(num_turns)) if num_turns[i] == 2
        ]
        three_turn_successes = [
            successes[i] for i in range(len(num_turns)) if num_turns[i] == 3
        ]

        block_dists = [x.block_dist for x in self.envs]

        avg_steps_to_goal = calc_avg_steps_to_goal(data.reward_history)

        median_steps_to_goal = calc_avg_steps_to_goal(
            data.reward_history, reduce_type="median"
        )

        # opt_path_sim = self._get_opt_path_sim(data)
        ood_report = self._construct_ood_report(data, successes=successes)

        report = MPCReport(
            success_rate=success_rate,
            success=successes,
            avg_steps_to_goal=avg_steps_to_goal,
            median_steps_to_goal=median_steps_to_goal,
            terminations=terminations,
            one_turn_success_rate=(
                sum(one_turn_successes) / len(one_turn_successes)
                if one_turn_successes
                else -1
            ),
            two_turn_success_rate=(
                sum(two_turn_successes) / len(two_turn_successes)
                if two_turn_successes
                else -1
            ),
            three_turn_success_rate=(
                sum(three_turn_successes) / len(three_turn_successes)
                if three_turn_successes
                else -1
            ),
            num_one_turns=len(one_turn_successes),
            num_two_turns=len(two_turn_successes),
            num_three_turns=len(three_turn_successes),
            num_turns=num_turns,
            block_dists=block_dists,
            ood_report=ood_report,
        )

        return report

    def evaluate(self):
        mpc_data = self._perform_mpc_in_chunks()

        report = self._construct_report(mpc_data)

        log_l1_planning_loss(result=mpc_data, prefix=self.prefix)

        mpc_data.targets = mpc_data.targets[:, :2]  # only keep (pos_x, pos_y)

        if self.config.visualize_planning:
            log_planning_plots(
                result=mpc_data,
                report=report,
                idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
                prefix=self.prefix,
                n_steps=self.config.n_steps,
                xy_action=True,
                plot_every=self.config.plot_every,
                quick_debug=self.quick_debug,
                pixel_mapper=self.pixel_mapper,
                plot_failure_only=self.config.plot_failure_only,
                log_pred_dist_every=self.config.log_pred_dist_every,
                mark_action=False,
            )

        return mpc_data, report
