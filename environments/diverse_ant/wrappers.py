import torch
from environments.ogbench.wrappers import NormEvalWrapper
from torchvision import transforms
from PIL import Image
import numpy as np


class DiverseAntNormEvalWrapper(NormEvalWrapper):
    def __init__(
        self,
        env,
        normalizer,
        stack_states=1,
        image_based: bool = True,
        init_ppos_idx: int = 2,
        image_size: tuple = (64, 64, 3),
    ):
        super().__init__(
            env=env,
            normalizer=normalizer,
            stack_states=stack_states,
            image_based=image_based,
            init_ppos_idx=init_ppos_idx,
        )
        self.image_transform = transforms.Resize(tuple(image_size[:2]))

    def step(self, action):
        """
        Step and return normalized obs
        """

        _, reward, done, truncated, info = self.env.step(action)
        observation = self.get_obs()
        info = self.get_info()

        return observation, reward, done, truncated, info

    def _resize_image(self, image_np):
        img = Image.fromarray(image_np)
        resized_img = self.image_transform(img)
        return np.array(resized_img)

    def _normalize_ob(self, ob):
        """
        Ob: (200, 200, 3)
        First downscales to 64x64. Then normalize.
        """

        ob = self._resize_image(ob.copy())
        ob = torch.from_numpy(ob).permute(2, 0, 1).float()
        ob = self.normalizer.normalize_state(ob)

        return ob

    def get_obs(self):
        ob = self.render()
        ob = self._normalize_ob(ob)
        return ob
