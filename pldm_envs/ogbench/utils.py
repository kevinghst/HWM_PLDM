import torch
import numpy as np
from PIL import Image


MAZE_STATS = {
    "antmaze-medium-v0": {
        "env_offset_x": 4,
        "env_offset_y": 4,
        "env_maze_unit": 4,
        "maze_map_grids": 8,
        "img_width": 81,
        "img_height": 81,
        "correction_factor": 0.3,
    },
    "ant_medium_diverse": {
        "env_offset_x": 4,
        "env_offset_y": 4,
        "env_maze_unit": 4,
        "maze_map_grids": 8,
        "img_width": 64,
        "img_height": 64,
        "correction_factor": 0.3,
    },
    "antmaze-u-v0": {
        "env_offset_x": 4,
        "env_offset_y": 4,
        "env_maze_unit": 4,
        "maze_map_grids": 5,
        "img_width": 81,
        "img_height": 81,
        "correction_factor": 0.7,
    },
}


class PixelMapper:
    def __init__(
        self,
        env_name: str,
        img_width: int = None,
        img_height: int = None,
        correction_factor: float = None,
    ):
        self.env_name = env_name

        if env_name in MAZE_STATS:
            env_key = env_name
        elif "medium" in env_name:
            env_key = "antmaze-medium-v0"
        else:
            raise NotImplementedError

        self.stats = MAZE_STATS[env_key]

        # override fields if given
        if img_width is not None:
            self.stats["img_width"] = img_width

        if img_height is not None:
            self.stats["img_height"] = img_height

        if correction_factor is not None:
            self.stats["correction_factor"] = correction_factor

        self.env_offset_x = self.stats["env_offset_x"]
        self.env_offset_y = self.stats["env_offset_y"]
        self.env_maze_unit = self.stats["env_maze_unit"]
        self.maze_map_grids = self.stats["maze_map_grids"]
        self.img_width = self.stats["img_width"]
        self.img_height = self.stats["img_height"]
        self.correction_factor = self.stats["correction_factor"]

    def obs_coord_to_pixel_coord(self, coord, flip_coord=True):
        env_offset_x = self.env_offset_x
        env_offset_y = self.env_offset_y
        env_maze_unit = self.env_maze_unit
        maze_map_grids = self.maze_map_grids
        img_width = self.img_width
        img_height = self.img_height
        correction_factor = self.correction_factor

        # Determine if coord is a torch tensor or numpy array
        is_tensor = isinstance(coord, torch.Tensor)

        # Define the range based on the maze's layout, offsets, and unit size
        x_min = -env_offset_x
        x_max = (maze_map_grids - 1) * env_maze_unit - env_offset_x
        y_min = -env_offset_y
        y_max = (maze_map_grids - 1) * env_maze_unit - env_offset_y

        # Reshape coord to (-1, 2) to handle arbitrary shapes
        original_shape = coord.shape[:-1]  # Shape except the last dimension
        coord_reshaped = coord.reshape(-1, 2) if is_tensor else coord.reshape(-1, 2)

        # Split x and y components
        x_obs, y_obs = coord_reshaped[:, 0], coord_reshaped[:, 1]

        # Normalize the coordinates to [0, 1] in the maze's bounds
        x_norm = (x_obs - x_min) / (x_max - x_min)
        y_norm = (y_obs - y_min) / (y_max - y_min)

        # Calculate the distance from the center of the image
        x_center, y_center = 0.5, 0.5
        dist_from_center = (
            torch.sqrt((x_norm - x_center) ** 2 + (y_norm - y_center) ** 2)
            if is_tensor
            else np.sqrt((x_norm - x_center) ** 2 + (y_norm - y_center) ** 2)
        )

        # Apply the correction based on distance from the center
        x_corrected = x_center + (x_norm - x_center) * (
            1 + correction_factor * dist_from_center
        )
        y_corrected = y_center + (y_norm - y_center) * (
            1 + correction_factor * dist_from_center
        )

        # Scale to pixel space
        x_pixel = (
            (x_corrected * img_width).to(torch.int32)
            if is_tensor
            else (x_corrected * img_width).astype(int)
        )
        y_pixel = (
            (y_corrected * img_height).to(torch.int32)
            if is_tensor
            else (y_corrected * img_height).astype(int)
        )

        # Calculate final coordinates
        if is_tensor:
            pixel_coords = torch.stack((img_width - y_pixel, x_pixel), dim=-1)
        else:
            pixel_coords = np.stack((img_width - y_pixel, x_pixel), axis=-1)

        # Reshape back to the original shape with (..., 2)

        output = (
            pixel_coords.reshape(*original_shape, 2)
            if is_tensor
            else pixel_coords.reshape(*original_shape, 2)
        )

        if flip_coord:
            output = output.flip(-1)

        return output


def test_pixel_mapper(env, samples=10):
    """
    Helper function to test whether pixel mapper is working correctly
    """

    obs = env.render()

    pixel_mapper = PixelMapper(
        env_name=env.name,
        img_width=obs.shape[0],
        img_height=obs.shape[1],
    )

    for i in range(samples):
        ob, _ = env.reset()
        obs = env.render().copy()
        xy = torch.from_numpy(ob[:2])
        pixel_xy = pixel_mapper.obs_coord_to_pixel_coord(xy, flip_coord=False)

        obs[pixel_xy[0], pixel_xy[1]] = [255, 0, 0]
        Image.fromarray(obs).save(f"pixel_mapper_test_{i}.png")
