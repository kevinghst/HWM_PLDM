from environments.diverse_maze.maze_draw import render_umaze
import numpy as np
import gym
import torch
import re
from environments.diverse_maze.data_generation.maze_stats import RENDER_STATS
from environments.diverse_maze.utils import *


def contains_wrapper(env, wrapper_class):
    """
    Recursively check if any wrapper in the nested structure is an instance of the specified class.

    Args:
        env: The Gym environment (with potential nested wrappers).
        wrapper_class: The wrapper class to check for.

    Returns:
        bool: True if the wrapper_class is found at any level, False otherwise.
    """
    if isinstance(env, wrapper_class):
        return True
    # Check the next level of wrapping if it exists
    if hasattr(env, "env"):
        return contains_wrapper(env.env, wrapper_class)
    return False


class NavigationWrapper(gym.Wrapper):
    """
    Functionalities
        converts xy to ij
        converts ij to xy
        can get subgoal xy to reach target
    """

    def __init__(self, env):
        super().__init__(env)
        self.map_layout = self.map_key.split("\\")

        def remove_suffix_if_matches(input_string):
            # Regular expression to match _* where * is an integer
            pattern = r"_(\d+)$"
            # Check if the string matches the pattern
            if re.search(pattern, input_string):
                # Remove the matching pattern from the end
                return re.sub(pattern, "", input_string)
            return input_string

        env_name_root = remove_suffix_if_matches(env.name)
        self.obs_min_total = RENDER_STATS[env_name_root]["obs_min_total"]
        self.obs_range_total = RENDER_STATS[env_name_root]["obs_range_total"]

    def sample_ij(self):
        # Collect all 'O' locations in the grid
        O_locations = [
            (i, j)
            for i, row in enumerate(self.map_layout)
            for j, char in enumerate(row)
            if char == "O"
        ]

        # Check if there are any 'O' locations
        if not O_locations:
            raise ValueError("No 'O' locations found in the grid.")

        # Randomly sample one of the 'O' locations
        return random.choice(O_locations)

    def sample_xy(self):
        rand_ij = self.sample_ij()
        return self.ij_to_xy(rand_ij)

    def edit_distance_btw_paths(self, path1, path2):
        """
        x, y each is a list of tuples (i, j) representing the path
        calcualtes edit distance between the two paths
        """
        m, n = len(path1), len(path2)
        # Create a DP table with dimensions (m+1) x (n+1)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize the base cases
        for i in range(m + 1):
            dp[i][0] = i  # Cost of deleting all characters from path1
        for j in range(n + 1):
            dp[0][j] = j  # Cost of inserting all characters into path1

        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if path1[i - 1] == path2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # No cost if characters match
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],  # Deletion
                        dp[i][j - 1],  # Insertion
                        dp[i - 1][j - 1],  # Substitution
                    )

        # The edit distance is in the bottom-right corner of the table
        return dp[m][n]

    def xy_to_ij(self, xy):
        return obs_to_ij(
            xy,
            obs_min_total=self.obs_min_total,
            obs_range_total=self.obs_range_total,
            n=len(self.map_layout),
        )

    def ij_to_xy(self, ij):
        return ij_to_obs(
            ij,
            obs_min_total=self.obs_min_total,
            obs_range_total=self.obs_range_total,
            n=len(self.map_layout),
        )

    def find_shortest_path(self, current_ij, target_ij):
        """
        Finds the shortest path between current_ij and target_ij in a grid.

        Args:
            current_ij (tuple): The starting point (i, j).
            target_ij (tuple): The target point (i, j).

        Returns:
            tuple: A tuple containing:
                - list of tuple: A list of (i, j) tuples representing the shortest path, including target_ij but excluding current_ij.
                - bool: True if there are multiple shortest paths, False otherwise.
        """
        map_layout = self.map_layout

        def is_valid(x, y):
            return (
                0 <= x < len(map_layout)
                and 0 <= y < len(map_layout[0])
                and map_layout[x][y] == "O"
            )

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        queue = deque(
            [(current_ij, [])]
        )  # Each element is (current_position, path_so_far)
        visited = set()
        visited.add(current_ij)

        shortest_paths_found = 0
        shortest_path_length = None
        result_path = []

        while queue:
            (current_i, current_j), path = queue.popleft()

            for di, dj in directions:
                next_i, next_j = current_i + di, current_j + dj

                if (next_i, next_j) == target_ij:
                    if shortest_path_length is None:
                        shortest_path_length = len(path) + 1
                        result_path = path + [(current_i, current_j), (next_i, next_j)]
                        shortest_paths_found += 1
                    elif len(path) + 1 == shortest_path_length:
                        shortest_paths_found += 1

                elif is_valid(next_i, next_j) and (next_i, next_j) not in visited:
                    visited.add((next_i, next_j))
                    queue.append(((next_i, next_j), path + [(current_i, current_j)]))

        has_multiple_shortest_paths = shortest_paths_found > 1
        return result_path[1:], has_multiple_shortest_paths

    def get_oracle_subgoal(self):
        """
        Given current state, returns the subgoal to reach the target.
        """

        # if current and target are in same cell, return target
        current_xy = self.unwrapped._get_obs()[:2]
        target_xy = self.get_target()
        target_ij = self.xy_to_ij(target_xy)

        current_ij = self.xy_to_ij(current_xy)
        if np.all(current_ij == target_ij):
            return target_xy

        # if not. use bfs to find the next cell in the shortest path to the target
        next_ij = self.find_shortest_path(current_ij, target_ij)[0][0]
        next_xy = self.ij_to_xy(next_ij)
        return next_xy


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=1, action_repeat_mode="null"):
        super().__init__(env)
        self.action_repeat = action_repeat
        self.action_repeat_mode = action_repeat_mode

    def get_obs(self, *args, **kwargs):
        obs = self.unwrapped._get_obs(*args, **kwargs)
        return obs

    def step(self, action):
        status = self.env.step(action)

        for i in range(1, self.action_repeat):
            if self.action_repeat_mode == "id":
                step_a = action
            elif self.action_repeat_mode == "linear":
                step_a = action - i * (action / self.action_repeat)
            elif self.action_repeat_mode == "null":
                step_a = np.array([0, 0])
            else:
                raise NotImplementedError

            status = self.env.step(step_a)

        return status


class NormEvalWrapper(gym.Wrapper):
    def __init__(self, env, normalizer=None, hybrid_obs=False, stack_states: int = 1):
        """
        hybrid_obs: if True, the observation is a dict of:
            {
                "image": torch.Tensor,
                "proprio": torch.Tensor,
            }
        otherwise, it is just the image.
        """
        self.normalizer = normalizer
        self.hybrid_obs = hybrid_obs
        self.stack_states = stack_states

        super().__init__(env)
        assert env.name.startswith("maze2d")

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        image = render_umaze(self.env, obs, normalizer=self.normalizer)

        if self.hybrid_obs:
            obs = {"image": image, "proprio": self.get_proprio_vel(normalized=True)}
        else:
            obs = image

        return obs

    def get_target_proprio(self):
        """
        return:
            numpy array (2,)

        For this environment we don't care about matching proprioceptive state during planning.
        So we just return zeros
        """
        return torch.zeros(2)

    def get_target_obs(self, return_stacked_states=True):
        goal = self.get_target()

        goal_image = render_umaze(
            self.env,
            np.concatenate([goal, np.zeros(2)]),  # append velocity component
            set_to_obs=False,  # TODO: DOUBLE CHECK
            normalizer=self.normalizer,
        )

        if return_stacked_states:
            goal_image = torch.cat([goal_image] * self.stack_states)

        if self.hybrid_obs:
            goal_obs = {
                "image": goal_image,
                "proprio": self.get_proprio_vel(normalized=True),
            }
        else:
            goal_obs = goal_image

        return goal_obs

    # alis for get_target_obs
    get_goal_rendered = get_target_obs

    def get_obs(self):
        obs = self.unwrapped._get_obs()
        image = render_umaze(self.env, obs, normalizer=self.normalizer)

        if self.hybrid_obs:
            obs = {"image": image, "proprio": self.get_proprio_vel(normalized=True)}
        else:
            obs = image

        return obs

    def get_proprio_vel(self, normalized=False):
        qvel = self.unwrapped._get_obs()[2:]
        qvel = torch.from_numpy(qvel).float()
        if self.normalizer is not None and normalized:
            qvel = self.normalizer.normalize_proprio_vel(qvel)
        return qvel

    get_proprio_vel = get_proprio_vel

    def get_info(self):
        return {
            "location": self.get_pos(),
            "proprio": self.get_proprio_vel(normalized=False),
        }

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        image = render_umaze(self.env, obs, normalizer=self.normalizer)

        if self.hybrid_obs:
            obs = {"image": image, "proprio": self.get_proprio_vel(normalized=True)}
        else:
            obs = image

        info = self.get_info()
        truncated = False

        return obs, rew, done, truncated, info

    def get_pos(self):
        obs = self.unwrapped._get_obs()
        return obs[:2]
