# copied from https://github.com/UM-ARM-Lab/pytorch_mppi

import logging
import time

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

# from planning.planner import Planner

logger = logging.getLogger(__name__)


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


class MPPI:
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(
        self,
        dynamics,
        running_cost,
        nx,
        noise_sigma,
        proprio_dim=0,
        num_samples=100,
        horizon=15,
        device="cpu",
        terminal_state_cost=None,
        lambda_=1.0,
        noise_mu=None,
        action_normalizer=None,
        u_init=None,
        U_init=None,
        u_scale=1,
        u_per_command=1,
        latent_actions=False,
        z_reg_coeff=0.1,
        step_dependent_dynamics=False,
        rollout_samples=1,
        var_samples=0,
        rollout_var_cost=0,
        rollout_obs_var_cost=0,
        rollout_proprio_var_cost=0,
        rollout_var_discount=0.95,
        sample_null_action=False,
        noise_abs_cost=False,
        cost_entity="obs_component",
    ):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for models that output a distribution)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        self.latent_actions = latent_actions
        self.z_reg_coeff = z_reg_coeff

        # dimensions of state and control
        self.nx = nx
        self.proprio_dim = proprio_dim
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.action_normalizer = action_normalizer
        self.u_scale = u_scale
        self.u_per_command = u_per_command

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(
            self.noise_mu, covariance_matrix=self.noise_sigma
        )
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.M = rollout_samples
        self.var_samples = var_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_obs_var_cost = rollout_obs_var_cost
        self.rollout_proprio_var_cost = rollout_proprio_var_cost
        self.rollout_var_discount = rollout_var_discount

        if self.var_samples:
            assert self.M == 1  # These should be mutually exclusive

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

        self.cost_entity = cost_entity

    # @handle_batch_input(n=2)
    def _dynamics(self, state, proprio, location, raw_location, u):
        return self.F(state, proprio, location, raw_location, u)

    # @handle_batch_input(n=2)
    def _running_cost(self, state, u, t):
        return (
            self.running_cost(state, u, t)
            if self.step_dependency
            else self.running_cost(state, u)
        )

    def _running_var_cost(self, states):
        """
        Input:
            states: (var_rollouts, num_samples, nx)
        Computes the sum of variance of the features across the var_rollouts
        Returns:
            (num_samples,)
        """
        var = torch.var(states, dim=0, unbiased=False)
        var_cost = var.sum(dim=-1)
        return var_cost

    def shift_nominal_trajectory(self):
        """
        Shift the nominal trajectory forward one step
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

    def command(
        self,
        state,
        proprio=None,
        location=None,
        raw_location=None,
        shift_nominal_trajectory=True,
    ):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :param shift_nominal_trajectory: Whether to roll the nominal trajectory forward one step. This should be True
        if the command is to be executed. If the nominal trajectory is to be refined then it should be False.
        :returns action: (nu) best action
        """
        if shift_nominal_trajectory:
            self.shift_nominal_trajectory()

        return self._command(state, proprio, location, raw_location)

    def _command(self, state, proprio, location, raw_location):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        self.proprio = (
            proprio.to(dtype=self.dtype, device=self.d) if proprio is not None else None
        )
        self.location = (
            location.to(dtype=self.dtype, device=self.d)
            if location is not None
            else None
        )
        self.raw_location = (
            raw_location.to(dtype=self.dtype, device=self.d)
            if raw_location is not None
            else None
        )
        cost_total = self._compute_total_cost_batch()
        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1.0 / eta) * self.cost_total_non_zero
        perturbations = []
        for t in range(self.T):
            perturbations.append(
                torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
            )
        perturbations = torch.stack(perturbations)
        self.U = self.U + perturbations
        if self.u_per_command == -1:
            # return all the actions
            return self.U
        action = self.U[: self.u_per_command]
        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]
        return action

    def change_horizon(self, horizon):
        if horizon < self.U.shape[0]:
            # truncate trajectory
            self.U = self.U[:horizon]
        elif horizon > self.U.shape[0]:
            # extend with u_init
            self.U = torch.cat(
                (self.U, self.u_init.repeat(horizon - self.U.shape[0], 1))
            )
        self.T = horizon

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _expand_input_for_ensemble(self, x, K, state_dim):
        if x is None:
            return None

        if x.shape != (K, state_dim):
            x = x.repeat(K, *[1 for _ in x.shape])

        # rollout action trajectory M times to estimate expected cost
        return x.repeat(self.M, *[1 for _ in x.shape])

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)
        cost_var_obs = torch.zeros_like(cost_total)
        cost_var_proprio = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        state = self._expand_input_for_ensemble(self.state, K=K, state_dim=self.nx)
        proprio = self._expand_input_for_ensemble(
            self.proprio, K=K, state_dim=self.proprio_dim
        )
        location = self._expand_input_for_ensemble(self.location, K=K, state_dim=2)
        raw_location = self._expand_input_for_ensemble(
            self.raw_location, K=K, state_dim=2
        )

        # states = []
        actions = []
        idx_start = (
            0
            if self.running_cost.sum_all_diffs
            else (
                T - self.running_cost.sum_last_n
                if self.running_cost.sum_last_n is not None
                else 0
            )
        )
        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t]

            dynamics_output = self._dynamics(
                state=state,
                proprio=proprio,
                location=location,
                raw_location=raw_location,
                u=u,
            )
            state = dynamics_output.ensemble_predictions
            proprio = dynamics_output.ensemble_proprio_component
            state_obs = dynamics_output.ensemble_obs_component
            location = dynamics_output.ensemble_location_component
            raw_locations = dynamics_output.ensemble_raw_locations

            if self.cost_entity == "obs_component":
                c = self._running_cost(state_obs, u, t)
            elif self.cost_entity == "proprio_component":
                c = self._running_cost(proprio, u, t)
            elif self.cost_entity == "location_component":
                c = self._running_cost(location, u, t)
            elif self.cost_entity == "raw_locations":
                c = self._running_cost(raw_locations, u, t)
            else:
                raise ValueError(f"cost_entity {self.cost_entity} not recognized.")

            if t >= idx_start:
                cost_samples = cost_samples + c

            if self.M > 1 and self.rollout_var_cost:
                # cost_var += c.var(dim=0) * (self.rollout_var_discount**t)
                # we're gonna do it differently - calcualte var from the representations
                vc = self._running_var_cost(state)
                cost_var += vc * (self.rollout_var_discount**t)

            if self.M > 1 and self.rollout_obs_var_cost:
                vc = self._running_var_cost(state_obs)
                cost_var_obs += vc * (self.rollout_var_discount**t)

            if self.M > 1 and self.rollout_proprio_var_cost:
                vc = self._running_var_cost(proprio)
                cost_var_proprio += vc * (self.rollout_var_discount**t)

            # Save total states/actions
            # states.append(state)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        # states = torch.stack(states, dim=-2)
        states = None

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples = cost_samples + c

        if self.latent_actions:
            # we assume we're regularizing towards a standard uniform prior for now
            prior_mus = torch.zeros_like(actions)
            prior_vars = torch.ones_like(actions)
            prior_d = torch.distributions.Normal(loc=prior_mus, scale=prior_vars)
            z_reg = -prior_d.log_prob(actions).mean(dim=(1, 2))
            z_reg = z_reg * self.z_reg_coeff

        cost_total = cost_total + cost_samples.mean(dim=0)
        cost_total = (
            cost_total
            + cost_var * self.rollout_var_cost
            + cost_var_obs * self.rollout_obs_var_cost
            + cost_var_proprio * self.rollout_proprio_var_cost
        )
        return cost_total, states, actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        noise = self.noise_dist.rsample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_action = self.U + noise
        if self.sample_null_action:
            perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(perturbed_action)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = (
                self.lambda_ * self.noise @ self.noise_sigma_inv
            )  # Like original paper

        rollout_cost, self.states, actions = self._compute_rollout_costs(
            self.perturbed_action
        )
        self.actions = actions / self.u_scale

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.action_normalizer is not None:
            for t in range(self.T):
                u = action[:, self._slice_control(t)]
                cu = self.action_normalizer(u)  # double check
                action[:, self._slice_control(t)] = cu
        return action

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    # def get_rollouts(self, state, num_rollouts=1):
    #     """
    #     :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
    #     :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
    #                          dynamics
    #     :returns states: num_rollouts x T x nx vector of trajectories

    #     """
    #     state = state.view(-1, self.nx)
    #     if state.size(0) == 1:
    #         state = state.repeat(num_rollouts, 1)

    #     T = self.U.shape[0]
    #     states = torch.zeros(
    #         (num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device
    #     )
    #     states[:, 0] = state
    #     for t in range(T):
    #         states[:, t + 1] = self._dynamics(
    #             states[:, t].view(num_rollouts, -1),
    #             self.u_scale * self.U[t].tile(num_rollouts, 1),
    #             t,
    #         )
    #     return states[:, 1:]


def run_mppi(
    mppi, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True
):
    dataset = torch.zeros(
        (retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d
    )
    total_reward = 0
    for i in range(iter):
        state = env.unwrapped.state.copy()
        command_start = time.perf_counter()
        action = mppi.command(state)
        elapsed = time.perf_counter() - command_start
        res = env.step(action.cpu().numpy())
        s, r = res[0], res[1]
        total_reward += r
        logger.debug(
            "action taken: %.4f cost received: %.4f time taken: %.5fs",
            action,
            -r,
            elapsed,
        )
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, : mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype)
        dataset[di, mppi.nx :] = action
    return total_reward, dataset
