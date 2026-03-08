import gym
import numpy as np


class OGBenchWrapper(gym.Wrapper):
    """
    To make API compatible with ogbench. Mainly used for data generation.
    """

    def __init__(self, env, min_state: np.ndarray, max_state: np.ndarray):
        """
        Args:
            min_state (2,) minimum (x,y) coord of env
            max_state (2,) maximum (x,y) coord of env
        """
        self.env = env
        self.min_state = min_state
        self.max_state = max_state
        self.effective_grid_size = len(env.maze_map) - 2  # excluding border
        self.box_dim = (max_state[0] - min_state[0]) / self.effective_grid_size
        super().__init__(env)

    def ij_to_xy(self, ij, margin=0.05):
        """
        Randomly sample a point in the grid cell.
        Note: may be different from ogbench class method
        """
        i, j = ij

        x_min = self.min_state[0] + i * self.box_dim
        x_max = x_min + self.box_dim
        y_min = self.min_state[1] + j * self.box_dim
        y_max = y_min + self.box_dim

        x = np.random.uniform(x_min + margin, x_max - margin)
        y = np.random.uniform(y_min + margin, y_max - margin)

        return np.array([x, y])

    def reset(self, options=None, *args, **kwargs):
        if "init_xy" in options and "goal_xy" in options:
            init_xy = options["init_xy"]
            goal_xy = options["goal_xy"]
        else:
            init_xy = self.ij_to_xy(options["task_info"]["init_ij"])
            goal_xy = self.ij_to_xy(options["task_info"]["goal_ij"])

        self.env.reset()
        self.env.set_state(qpos=init_xy, qvel=np.array([0, 0]))
        self.env.set_target(goal_xy)

        return ob, info

    def _get_obs(self, *args, **kwargs):
        obs = self.env._get_obs(*args, **kwargs)
        return obs

    def step(self, action):
        status = self.env.step(action)

        return status
