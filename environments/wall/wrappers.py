import gym


class NormEvalWrapper(gym.Wrapper):
    def __init__(self, env, normalizer=None):
        super().__init__(env)
        self.normalizer = normalizer

    def get_info(self):
        return {
            "location": self.get_pos(),
        }

    def get_target(self):
        return self.env.target_position.cpu().numpy()

    def get_target_obs(self):
        target_obs = self.env.get_target_obs()
        if self.normalizer is not None:
            target_obs = self.normalizer.normalize_state(target_obs)
        return target_obs

    def get_pos(self):
        return self.env.dot_position.cpu().numpy()

    def get_obs(self):
        obs = self.env._get_obs()
        if self.normalizer is not None:
            obs = self.normalizer.normalize_state(obs)
        return obs

    def reset(self):
        obs, *_ = self.env.reset()
        if self.normalizer is not None:
            obs = self.normalizer.normalize_state(obs)
        return obs

    def step(self, action):
        obs, reward, done, truncated, _ = self.env.step(action)
        if self.normalizer is not None:
            obs = self.normalizer.normalize_state(obs)

        info = self.get_info()

        return obs.cpu().numpy(), reward, done, truncated, info
