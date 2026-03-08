from gymnasium.core import Wrapper
import torch


class NormEvalWrapper(Wrapper):
    """
    Return normalized observations. And other useful calls for planning.
    Can be used for both pixel based and state based envs.
    """

    def __init__(
        self,
        env,
        normalizer,
        stack_states: int,
        image_based: bool = False,
        init_ppos_idx: int = 2,
    ):
        super().__init__(env)
        self.normalizer = normalizer
        self.image_based = image_based or (env.unwrapped._ob_type == "pixels")
        self.stack_states = stack_states
        self.init_ppos_idx = init_ppos_idx
        self._target_visual = None

    def _normalize_ob(self, ob):
        if self.image_based:
            ob = torch.from_numpy(ob.copy()).permute(2, 0, 1).float()
            ob = self.normalizer.normalize_state(ob)
        else:
            ob = torch.from_numpy(ob.copy()).float()
            ob = self.normalizer.normalize_state(ob[:2])

        return ob

    def get_goal_xy(self, normalized=False):
        """
        return:
            numpy array (2,)
        """
        goal_xy = self.env.unwrapped.cur_goal_xy
        goal_xy = torch.from_numpy(goal_xy.copy()).float()
        if normalized:
            goal_xy = self.normalizer.normalize_location(goal_xy)
        return goal_xy

    def reset(self, **kwargs):
        """
        Reset and return normalized obs
        """

        observation, info = self.env.reset(**kwargs)
        if self.image_based:
            info["goal_rendered"] = self._normalize_ob(info["goal_rendered"])

        return observation, info

    def reset_exact(self, **kwargs):
        """
        Reset exact and return normalized obs
        """

        observation, info = self.env.reset_exact(**kwargs)
        if self.image_based:
            info["goal_rendered"] = self._normalize_ob(info["goal_rendered"])
        return observation, info

    def get_info(self):
        return {
            "location": self.get_pos(),
            "qpos": self.get_qpos(),
            "proprio": self.get_proprio(),
        }

    def step(self, action):
        """
        Step and return normalized obs
        """

        observation, reward, done, truncated, info = self.env.step(action)
        observation = self._normalize_ob(observation)

        info = self.get_info()

        return observation, reward, done, truncated, info

    # def render(self, **kwargs):
    #     # Custom rendering logic
    #     print("Custom rendering logic")
    #     return self.env.render(**kwargs)

    def get_target(self):
        return self.unwrapped.cur_goal_xy

    def get_obs(self):
        """
        return:
            normalized observations
            torch float (ch, w, h) or (d)
        """
        ob = self.env.unwrapped.get_ob()
        ob = self._normalize_ob(ob)

        return ob

    def get_target_obs(self, return_stacked_states=True):
        if self.image_based:
            goal_obs = self.env.unwrapped.goal_rendered
            goal_obs = self._normalize_ob(goal_obs)
            if return_stacked_states:
                goal_obs = torch.cat([goal_obs] * self.stack_states)
        else:
            goal_obs = self.get_goal_xy(normalized=True)

        return goal_obs

    def get_target_proprio(self):
        """
        return:
            numpy array (27,)

        For this environment we don't care about matching proprioceptive state during planning.
        So we just return zeros
        """
        return torch.zeros(27)

    def get_proprio(self, normalized=False):
        """
        return:
            numpy array (27,) or (29,)
        """
        ob = self.env.unwrapped.get_ob(ob_type="states")

        if normalized:
            ob = torch.from_numpy(ob.copy()).float()
            ppos = self.normalizer.normalize_proprio_pos(ob[self.init_ppos_idx : 15])
            pvel = self.normalizer.normalize_proprio_vel(ob[15:])
            ob = torch.cat([ppos, pvel], dim=0)
        else:
            ob = torch.from_numpy(ob[self.init_ppos_idx :]).float()

        return ob

    def get_pos(self, normalized=False):
        """
        return:
            numpy array (2,) xy location
        """
        ob = self.env.unwrapped.get_ob(ob_type="states")[:2]
        ob = torch.from_numpy(ob.copy()).float()
        if normalized:
            ob = self.normalizer.normalize_location(ob)

        return ob

    def get_proprio_vel(self, normalized=False):
        ob = self.env.unwrapped.get_ob(ob_type="states")
        qvel = ob[-14:]

        if normalized:
            qvel = torch.from_numpy(qvel.copy()).float()
            qvel = self.normalizer.normalize_proprio_vel(qvel)

        return qvel

    def get_qpos(self):
        """
        Return proprio pos
        """
        ob = self.env.unwrapped.get_ob(ob_type="states")
        qpos = ob[:15]
        return qpos

    def get_proprio_pos(self, normalized=False):
        """
        Return proprio pos. WITHOUT global xy
        """
        ob = self.env.unwrapped.get_ob(ob_type="states")
        proprio_pos = ob[self.init_ppos_idx : 15]

        if normalized:
            proprio_pos = torch.from_numpy(proprio_pos.copy()).float()
            proprio_pos = self.normalizer.normalize_proprio_pos(proprio_pos)

        return proprio_pos
