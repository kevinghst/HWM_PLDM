from pldm.planning import objectives_v2
import torch
from pldm.models.hjepa import HJEPA
from pldm_envs.utils.normalizer import Normalizer
from pldm.planning.planners.enums import PlannerType
from pldm.planning.planners.mppi_planner import MPPIPlanner
from pldm.planning.planners.two_lvl_planner import TwoLvlPlanner
from pldm.planning.utils import normalize_actions
from abc import ABC
from pldm.planning.enums import MPCResult, PooledMPCResult
import numpy as np
from pldm.models.utils import flatten_conv_output
from tqdm import tqdm
from pldm.logger import Logger
from pldm.models.utils import flatten_ensemble_conv_output


class MPCEvaluator(ABC):
    def __init__(
        self,
        config,
        model: HJEPA,
        prober: torch.nn.Module,
        normalizer: Normalizer,
        quick_debug: bool = False,
        prefix: str = "",
        pixel_mapper=None,
        image_based=True,
        hierarchical=False,
        env_name=None,
        l2_use_latent_mean_std: bool = False,
    ):
        self.config = config
        self.model = model
        self.prober = prober
        self.normalizer = normalizer
        self.quick_debug = quick_debug
        self.prefix = prefix
        self.pixel_mapper = pixel_mapper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_based = image_based
        self.hierarchical = hierarchical
        self.l2_use_latent_mean_std = l2_use_latent_mean_std

    def close(self):
        pass

    def _infer_chunk_sizes(self):
        config = self.config

        if config.level1.planner_type == PlannerType.MPPI:
            # n_envs_batch_size = 500000 // (config.num_samples * config.max_plan_length)
            n_envs_batch_size = config.n_envs_batch_size
        else:
            n_envs_batch_size = config.n_envs

        chunk_sizes = [n_envs_batch_size] * (config.n_envs // n_envs_batch_size) + (
            [config.n_envs % n_envs_batch_size]
            if config.n_envs % n_envs_batch_size != 0
            else []
        )

        return chunk_sizes

    def _construct_h_planner(self, n_envs: int):
        l1_planner = self._construct_planner(n_envs=n_envs, l1_to_l2=True)

        l2_planner = self._construct_planner(
            n_envs=n_envs,
            l2=True,
        )

        h_planner = TwoLvlPlanner(
            l1_planner=l1_planner,
            l2_planner=l2_planner,
            l2_step_skip=self.model.config.step_skip,
        )

        return h_planner

    def _construct_planner(
        self,
        n_envs: int,
        l2: bool = False,
        l1_to_l2: bool = False,
    ):
        """
        Params:
            n_envs: how many environments to evaluate in parallel
            l2: whether to construct a level 2 planner
            l1_to_l2: whether to construct a level 1 planner for hierarchical planning
        """
        latent_actions = l2 and self.model.level2.config.predictor.z_dim > 0

        if l2:
            config = self.config.level2
            model = self.model.level2
            prober = self.prober_l2
            objective = objectives_v2.ReprTargetMPCObjective2(
                model=model,
                sum_all_diffs=config.sum_all_diffs,
                sum_last_n=config.sum_last_n,
            )
        else:
            config = self.config.level1
            model = self.model.level1
            prober = self.prober
            objective = objectives_v2.ReprTargetMPCObjective(
                model=model,
                proprio_cost=config.proprio_cost,
                sum_all_diffs=config.sum_all_diffs,
                loss_coeff_first=config.loss_coeff_first,
                loss_coeff_last=config.loss_coeff_last,
                pred_encoder=self.model.level2.backbone if l1_to_l2 else None,
            )

        dynamic_bounds = None
        if (
            l2
            and latent_actions
            and hasattr(self.normalizer, "l2_latent_min_bounds")
            and self.normalizer.l2_latent_min_bounds is not None
        ):
            dynamic_bounds = {
                "min_bounds": self.normalizer.l2_latent_min_bounds,
                "max_bounds": self.normalizer.l2_latent_max_bounds,
            }

        action_normalizer = lambda x: normalize_actions(
            x,
            min_norm=config.min_step,
            max_norm=config.max_step,
            clamp_actions=latent_actions or config.clamp_actions,
            env_name=self.config.env_name,
            dynamic_bounds=dynamic_bounds,
        )

        if config.planner_type == PlannerType.MPPI:
            planner = MPPIPlanner(
                config.mppi,
                model=model,
                normalizer=self.normalizer,
                objective=objective,  # [300, 13456]
                prober=prober,
                action_normalizer=action_normalizer,
                n_envs=n_envs,
                l2=l2,
                projected_cost=config.projected_cost,
                cost_entity=config.cost_entity,
                cost_dim_range=config.cost_dim_range,
                use_latent_mean_std=self.l2_use_latent_mean_std,
            )
        else:
            raise NotImplementedError(f"Unknown planner type {config.planner_type}")

        return planner

    def _perform_mpc_in_chunks(self):
        """
        Divide it up in chunks in order to prevent OOM
        """
        chunk_sizes = self._infer_chunk_sizes()

        mpc_data = PooledMPCResult()
        chunk_offset = 0

        for chunk_size in chunk_sizes:
            envs = self.envs[chunk_offset : chunk_offset + chunk_size]
            planner = self._construct_planner(n_envs=chunk_size)

            if self.hierarchical:
                h_planner = self._construct_h_planner(n_envs=chunk_size)
                mpc_result = self._perform_h_mpc(
                    h_planner=h_planner,
                    planner=planner,
                    envs=envs,
                )
            else:
                mpc_result = self._perform_mpc(
                    planner=planner,
                    envs=envs,
                )

            mpc_data.observations.append(mpc_result.observations)
            mpc_data.locations.append(mpc_result.locations)
            mpc_data.action_history.append(mpc_result.action_history)
            mpc_data.reward_history.append(mpc_result.reward_history)
            mpc_data.pred_locations.append(mpc_result.pred_locations)
            mpc_data.final_preds_dist.append(mpc_result.final_preds_dist)
            mpc_data.targets.append(mpc_result.targets)
            mpc_data.loss_history.append(mpc_result.loss_history)
            mpc_data.qpos_history.append(mpc_result.qpos_history)
            mpc_data.proprio_history.append(mpc_result.proprio_history)
            mpc_data.ensemble_var_history.append(mpc_result.ensemble_var_history)
            mpc_data.ensemble_obs_var_history.append(
                mpc_result.ensemble_obs_var_history
            )
            mpc_data.ensemble_proprio_var_history.append(
                mpc_result.ensemble_proprio_var_history
            )
            mpc_data.success_history.append(mpc_result.success_history)
            mpc_data.visual_observations.append(mpc_result.visual_observations)
            mpc_data.visual_targets.append(mpc_result.visual_targets)

            # for hierarchy
            if self.hierarchical:
                mpc_data.pred_locations_l2.append(mpc_result.pred_locations_l2)
                mpc_data.loss_history_l2.append(mpc_result.loss_history_l2)

            chunk_offset += chunk_size

        mpc_data.concatenate_chunks()

        return mpc_data

    def _perform_h_mpc(self, h_planner, planner, envs):
        """
        Hierarchical planning.
        Two stages:
        1. Perform MPC with the hierarchical planner
        2. Perform MPC with the level 1 planner for last part of the trip.
        Rationale: Hierarchical MPC might have some issue with fine-grained control.
            Could be problematic for the final part of the job.
        """
        # Stage 1
        mpc_result_1 = self._perform_mpc(
            planner=h_planner,
            envs=envs,
            bilevel_planning=True,
        )

        # Stage 2: Use flat L1 planning for the final steps near the goal
        if self.config.final_trans_steps:
            mpc_result_2 = self._perform_mpc(
                planner=planner,
                envs=envs,
                max_steps_override=self.config.final_trans_steps,
            )
            # combine results
            mpc_result = MPCResult(
                observations=mpc_result_1.observations
                + mpc_result_2.observations[1:],  #
                locations=mpc_result_1.locations + mpc_result_2.locations[1:],  #
                action_history=mpc_result_1.action_history
                + mpc_result_2.action_history,  #
                reward_history=mpc_result_1.reward_history
                + mpc_result_2.reward_history,  #
                pred_locations=mpc_result_1.pred_locations
                + mpc_result_2.pred_locations,  #
                final_preds_dist=mpc_result_1.final_preds_dist,
                targets=mpc_result_1.targets,
                loss_history=mpc_result_1.loss_history,
                ensemble_var_history=mpc_result_1.ensemble_var_history,
                ensemble_obs_var_history=mpc_result_1.ensemble_obs_var_history,
                ensemble_proprio_var_history=mpc_result_1.ensemble_proprio_var_history,
                qpos_history=mpc_result_1.qpos_history + mpc_result_2.qpos_history[1:],
                proprio_history=mpc_result_1.proprio_history
                + mpc_result_2.proprio_history[1:],
                # for hierarchy
                pred_locations_l2=mpc_result_1.pred_locations_l2,
                loss_history_l2=mpc_result_1.loss_history_l2,
                success_history=mpc_result_1.success_history
                + mpc_result_2.success_history[1:],
                visual_observations=mpc_result_1.visual_observations
                + mpc_result_2.visual_observations[1:],
                visual_targets=mpc_result_1.visual_targets,
            )
        else:
            mpc_result = mpc_result_1

        return mpc_result

    def _encode_targets(self, envs, bilevel_planning=False, squeeze_target=True):
        targets_obs = torch.stack([e.get_target_obs() for e in envs]).to(self.device)

        # encode target obs
        if self.model.level1.backbone.using_proprio:
            proprio_states = torch.stack([e.get_target_proprio() for e in envs]).to(
                self.device
            )
        else:
            proprio_states = None

        if self.model.level1.backbone.using_location:
            locations = torch.stack([e.get_goal_xy(normalized=True) for e in envs]).to(
                self.device
            )
        else:
            locations = None

        l1_output = self.model.level1.backbone(
            targets_obs, proprio=proprio_states, locations=locations
        )

        target_obs = l1_output.obs_component.detach()
        target_proprio = l1_output.proprio_component
        target_locations = l1_output.location_component
        target_raw_locations = l1_output.raw_locations

        if target_proprio is not None:
            target_proprio = target_proprio.detach()

        if not bilevel_planning:
            if self.config.level1.cost_entity == "obs_component":
                if len(target_obs.shape) == 3:
                    target_obs = target_obs.unsqueeze(0)
                target_obs_flat = flatten_conv_output(target_obs)
                if target_obs_flat.shape[0] == 1 and squeeze_target:
                    target_obs_flat = target_obs_flat.squeeze(0)
                return target_obs_flat
            elif self.config.level1.cost_entity == "proprio_component":
                return target_proprio
            elif self.config.level1.cost_entity == "location_component":
                return target_locations
            elif self.config.level1.cost_entity == "raw_locations":
                return target_raw_locations
            else:
                raise NotImplementedError(
                    f"Unknown cost entity {self.config.level1.cost_entity}"
                )

        l2_output = self.model.level2.backbone(target_obs, proprio=target_proprio)

        target_obs = l2_output.obs_component.detach()
        target_proprio = l2_output.proprio_component
        if target_proprio is not None:
            target_proprio = target_proprio.detach()

        if self.config.level2.cost_entity == "obs_component":
            if len(target_obs.shape) == 3:
                target_obs = target_obs.unsqueeze(0)
            target_obs_flat = flatten_conv_output(target_obs)
            if target_obs_flat.shape[0] == 1 and squeeze_target:
                target_obs_flat = target_obs_flat.squeeze(0)
            return target_obs_flat
        elif self.config.level2.cost_entity == "proprio_component":
            return target_proprio
        else:
            raise NotImplementedError(
                f"Unknown cost entity {self.config.level2.cost_entity}"
            )

    def _perform_mpc(
        self,
        planner,
        envs,
        bilevel_planning: bool = False,
        max_steps_override: int = None,
    ):
        """
        Parameters:
            starts: (bs, 4)
            targets: (bs, 4)
            max_steps_override: if provided, use this as the max number of steps
                instead of computing from config. Used for Stage 2 of hierarchical
                planning where we only want to run for final_trans_steps.
        Outputs:
            observations: list of a_T (bs, 3, 64, 64) or (bs, 2)
            locations: list of a_T (bs, 2)
            action_history: list of a_T (bs, p_T, 2)
            reward_history: list of a_T (bs,)
            pred_locations: list of a_T (p_T, bs, 1, 2)
            targets: (bs, 4)
            loss_history: list of a_T (n_iters,)
        """
        orig_training_state = self.model.training

        targets = [e.get_target() for e in envs]
        targets = torch.from_numpy(np.stack(targets))

        if not self.image_based and envs[0]._target_visual is not None:
            visual_targets = [e.get_target_visual() for e in envs]
            visual_targets = torch.from_numpy(np.stack(visual_targets))
        else:
            visual_targets = None

        targets_t = self._encode_targets(envs, bilevel_planning=bilevel_planning, squeeze_target=False)
        planner.reset_targets(targets_t, repr_input=True)

        observation_history = [torch.stack([e.get_obs() for e in envs])]

        obs_t = observation_history[0]
        if self.image_based:
            obs_t = torch.cat([obs_t] * self.config.stack_states, dim=1)  # VERIFY

        action_history = []
        reward_history = []
        location_history = []
        qpos_history = []
        proprio_history = []
        success_history = []

        visual_observations = []
        pred_locations_history = []
        loss_history = []
        final_preds_dist_history = []
        ensemble_var_history = []
        ensemble_obs_var_history = []
        ensemble_proprio_var_history = []

        # for hierarchy
        pred_locations_l2_history = []
        loss_history_l2 = []

        init_infos = [e.get_info() for e in envs]
        if "location" in init_infos[0]:
            location_history.append(np.array([info["location"] for info in init_infos]))

        if "qpos" in init_infos[0]:
            qpos_history.append(np.array([info["qpos"] for info in init_infos]))

        if "proprio" in init_infos[0]:
            proprio_history.append(np.array([info["proprio"] for info in init_infos]))

        if "visual_obs" in init_infos[0]:
            visual_observations.append(
                np.array([info["visual_obs"] for info in init_infos])
            )

        if max_steps_override is not None:
            # Used for Stage 2 of hierarchical planning (final_trans_steps)
            max_steps = max_steps_override
        elif bilevel_planning:
            step_skip = self.model.config.step_skip
            max_plan_horizon_l2 = self.config.level2.max_plan_length * step_skip
            # For hierarchical planning, reserve final_trans_steps for Stage 2 (flat L1)
            # so that total steps = Stage 1 + Stage 2 = n_steps
            final_trans_steps = getattr(self.config, "final_trans_steps", 0)
            n_steps_stage1 = self.config.n_steps - final_trans_steps
            max_steps = min(n_steps_stage1, max_plan_horizon_l2)
        else:
            max_steps = self.config.n_steps

        for i in tqdm(range(max_steps), desc="Planning steps"):
            if i % self.config.replan_every == 0:
                if self.model.level1.using_proprio_pos:
                    curr_proprio_pos = [
                        e.get_proprio_pos(normalized=True) for e in envs
                    ]
                    curr_proprio_pos = torch.from_numpy(
                        np.stack(curr_proprio_pos)
                    ).float()
                else:
                    curr_proprio_pos = None

                if self.model.level1.using_proprio_vel:
                    curr_proprio_vel = [
                        e.get_proprio_vel(normalized=True) for e in envs
                    ]
                    curr_proprio_vel = torch.from_numpy(
                        np.stack(curr_proprio_vel)
                    ).float()
                else:
                    curr_proprio_vel = None

                if self.model.level1.using_location:
                    curr_locations = [e.get_pos(normalized=True) for e in envs]
                    curr_locations = torch.from_numpy(np.stack(curr_locations)).float()
                else:
                    curr_locations = None

                if bilevel_planning:
                    plan_size = (max_plan_horizon_l2 - i + step_skip - 1) // step_skip
                    plan_size = max(plan_size, self.config.level2.min_plan_length)
                else:
                    plan_size = min(
                        self.config.n_steps - i, self.config.level1.max_plan_length
                    )

                planning_result = planner.plan(
                    obs_t,
                    curr_proprio_pos=curr_proprio_pos,
                    curr_proprio_vel=curr_proprio_vel,
                    curr_locations=curr_locations,
                    plan_size=plan_size,
                    repr_input=False,
                )

                if bilevel_planning:
                    planning_result_l2 = planning_result.level2
                    planning_result = planning_result.level1

            if bilevel_planning:
                pred = self._get_relevant_pred(
                    planning_result_l2,
                    self.config.level2.cost_entity,
                )
            else:
                pred = self._get_relevant_pred(
                    planning_result,
                    self.config.level1.cost_entity,
                )
            pred_dist = torch.norm(pred - targets_t.unsqueeze(0), dim=2).cpu()
            final_preds_dist_history.append(pred_dist)

            # calculate ensemble variance
            ensemble_predictions = planning_result.ensemble_predictions
            if ensemble_predictions.shape[1] > 1:
                ensemble_var = self._calculate_ensemble_var(ensemble_predictions.cpu())
                ensemble_var_history.append(ensemble_var)

            ensemble_obs_component = planning_result.ensemble_obs_component
            if ensemble_obs_component.shape[1] > 1:
                ensemble_obs_var = self._calculate_ensemble_var(
                    ensemble_obs_component.cpu()
                )
                ensemble_obs_var_history.append(ensemble_obs_var)

            ensemble_proprio_component = planning_result.ensemble_proprio_component
            if (
                ensemble_proprio_component is not None
                and ensemble_proprio_component.shape[1] > 1
            ):
                ensemble_proprio_var = self._calculate_ensemble_var(
                    ensemble_proprio_component.cpu()
                )
                ensemble_proprio_var_history.append(ensemble_proprio_var)

            planned_actions = (
                planning_result.actions[:, i % self.config.replan_every :]
                .detach()
                .cpu()
            )

            if self.config.random_actions:
                results = [
                    envs[j].step(envs[0].action_space.sample())
                    for j in range(len(envs))
                ]
            else:
                results = [
                    envs[j].step(
                        planned_actions[j, 0].detach().cpu().contiguous().numpy()
                    )
                    for j in range(len(envs))
                ]

            assert len(results[0]) == 5
            current_obs = torch.from_numpy(np.stack([r[0] for r in results])).float()
            rewards_t = torch.from_numpy(np.stack([r[1] for r in results])).float()
            infos = [r[4] for r in results]

            action_history.append(planned_actions.detach().cpu())
            observation_history.append(current_obs)
            reward_history.append(rewards_t)

            if "location" in infos[0]:
                location_history.append(np.array([info["location"] for info in infos]))

            if "qpos" in infos[0]:
                qpos_history.append(np.array([info["qpos"] for info in infos]))

            if "proprio" in infos[0]:
                proprio_history.append(np.array([info["proprio"] for info in infos]))

            if "success" in infos[0]:
                success_history.append(np.array([info["success"] for info in infos]))

            if "visual_obs" in infos[0] and visual_observations is not None:
                visual_observations.append(
                    np.array([info["visual_obs"] for info in infos])
                )

            if planning_result.locations is not None:
                pred_locations = planning_result.locations.detach().cpu()
                pred_locations = pred_locations.squeeze(2)
                pred_locations_history.append(pred_locations)

            # stack states if necessary for next iteration
            if self.config.stack_states == 1:
                obs_t = current_obs
            else:
                obs_t = torch.cat(
                    [obs_t[:, current_obs.shape[1] :], current_obs], dim=1
                )

            loss_history.append(planning_result.losses)

            if bilevel_planning:
                if planning_result_l2.locations is not None:
                    pred_locations_l2 = planning_result_l2.locations.detach().cpu()
                    if pred_locations_l2.dim() > 2 and pred_locations_l2.shape[2] == 1:
                        pred_locations_l2 = pred_locations_l2.squeeze(2)
                    pred_locations_l2_history.append(pred_locations_l2)
                loss_history_l2.append(planning_result_l2.losses)

        observation_history = [
            self.normalizer.unnormalize_state(o) for o in observation_history
        ]

        self.model.train(orig_training_state)

        # put everything on cpu
        observation_history = [o.cpu() for o in observation_history]

        return MPCResult(
            observations=observation_history,
            locations=[torch.from_numpy(x) for x in location_history],
            action_history=action_history,
            reward_history=reward_history,
            pred_locations=pred_locations_history,
            final_preds_dist=final_preds_dist_history,
            targets=targets,
            loss_history=loss_history,
            ensemble_var_history=ensemble_var_history,
            ensemble_obs_var_history=ensemble_obs_var_history,
            ensemble_proprio_var_history=ensemble_proprio_var_history,
            qpos_history=[torch.from_numpy(x) for x in qpos_history],
            proprio_history=[torch.from_numpy(x) for x in proprio_history],
            # for hierarchy
            pred_locations_l2=pred_locations_l2_history,
            loss_history_l2=loss_history_l2,
            success_history=[torch.from_numpy(x) for x in success_history],
            visual_observations=[torch.from_numpy(x) for x in visual_observations],
            visual_targets=visual_targets,
        )

    def _get_relevant_pred(self, planning_result, cost_entity):
        """
        Get the relevant prediction to diff against the target
        """
        if cost_entity == "obs_component":
            pred = planning_result.pred_obs
            pred = flatten_conv_output(pred)
        elif cost_entity == "proprio_component":
            pred = planning_result.pred_proprio
        elif cost_entity == "location_component":
            pred = planning_result.pred_location
        elif cost_entity == "raw_locations":
            pred = planning_result.raw_locations
        else:
            raise NotImplementedError(f"Unknown cost entity {cost_entity}")
        return pred

    def _calculate_ensemble_var(self, ensemble_states):
        ensemble_states = flatten_ensemble_conv_output(ensemble_states)
        if ensemble_states.shape[1] > 1:
            ensemble_var = torch.var(ensemble_states, dim=1).sum(-1)  # (pred_T, bs)
        return ensemble_var

    def _log_ensemble_var(self, ensemble_var_history, plot_prefix="", var_type="state"):
        """
        Plot ensemble variance across predicted timesteps for different action steps
        Action step 0
        Action step N * 1/3
        Action step N * 2/3
        """

        steps = [0, len(ensemble_var_history) // 3, len(ensemble_var_history) * 2 // 3]

        for step in steps:
            var = ensemble_var_history[step]
            # average across batch
            var = var.mean(dim=1)

            Logger.run().log_line_plot(
                data=[[i, x.item()] for i, x in enumerate(var)],
                plot_name=f"{plot_prefix}_planning_step_{step}_ensemble_{var_type}_var",
            )

        # then plot the aggregate variance across the steps
        last_step = steps[-1]
        # make sure all the ensemble variances have the same length
        last_step_pred_len = ensemble_var_history[last_step].shape[0]
        aggregate_vars = [
            var[:last_step_pred_len].mean() for var in ensemble_var_history[:last_step]
        ]
        Logger.run().log_line_plot(
            data=[[i, x.item()] for i, x in enumerate(aggregate_vars)],
            plot_name=f"{plot_prefix}_planning_aggregate_ensemble_var",
        )
