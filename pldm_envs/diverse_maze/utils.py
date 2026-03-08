import torch
from pldm_envs.diverse_maze.data_generation.maze_stats import RENDER_STATS
from typing import *
import numpy as np

import torchvision.transforms as transforms
from collections import deque
import random


class PixelMapper:
    def __init__(self, env_name):
        self.env_name = env_name
        if "diverse" in env_name:
            key_word = "diverse"
            index = env_name.find(key_word)
            if index != -1:
                env_name = env_name[: index + len(key_word)]

        self.stats = RENDER_STATS[env_name]
        self.image_topleft_in_obs_coord = torch.as_tensor(
            self.stats["image_topleft_in_obs_coord"]
        )
        self.scale_coord_obs_to_pixel = self.stats["scale_coord_obs_to_pixel"]
        self.transformed = self.stats["transformed"]
        self.crop_left_top = self.stats["crop_left_top"]
        self.scale_factor = self.stats["scale_factor"]

    def obs_coord_to_pixel_coord_v2(
        self,
        coord: Union[torch.Tensor, Tuple[float, float]],
        flip_coord=True,
        image_width=None,
    ) -> torch.Tensor:
        """
        flip coord is needed when you plot in matplotlib...
        """
        coord = torch.as_tensor(coord)
        original_shape = coord.shape

        # Ensure coord is 2D (n, 2) even if a single (2,) is passed
        if coord.ndim == 1:
            coord = coord.unsqueeze(0)
        else:
            coord = coord.view(-1, 2)

        obs_min_total = self.stats["obs_min_total"]
        obs_range_total = self.stats["obs_range_total"]
        if image_width is None:
            image_width = self.stats["image_width"]

        jj, ii = ((coord - obs_min_total) / obs_range_total * image_width).unbind(-1)

        pixel_coord = torch.stack([image_width - ii, jj], dim=-1)

        if original_shape == (2,):
            pixel_coord = pixel_coord.squeeze(0)
        else:
            pixel_coord = pixel_coord.view(*original_shape)

        if flip_coord:
            pixel_coord = pixel_coord.flip(-1)

        return pixel_coord

    def obs_coord_to_pixel_coord(
        self, coord: Union[torch.Tensor, Tuple[float, float]], flip_coord=True
    ) -> torch.Tensor:
        """
        flip coord is needed when you plot in matplotlib...
        """

        return self.obs_coord_to_pixel_coord_v2(coord, flip_coord)

        coord = torch.as_tensor(coord)
        original_shape = coord.shape

        # Ensure coord is 2D (n, 2) even if a single (2,) is passed
        if coord.ndim == 1:
            coord = coord.unsqueeze(0)  # Convert (2,) to (1, 2)
        else:
            coord = coord.view(-1, 2)

        jj, ii = (
            (coord - self.image_topleft_in_obs_coord)  # (-1.95, 8.55)
            .mul(self.scale_coord_obs_to_pixel)  # (500 / 10.5)
            .unbind(-1)
        )

        pixel_coord = torch.stack([-ii, jj], dim=-1)

        if self.transformed:
            pixel_coord[:, 0] = (
                pixel_coord[:, 0] - self.crop_left_top[0]
            ) / self.scale_factor
            pixel_coord[:, 1] = (
                pixel_coord[:, 1] - self.crop_left_top[1]
            ) / self.scale_factor

        if original_shape == (2,):
            pixel_coord = pixel_coord.squeeze(0)
        else:
            pixel_coord = pixel_coord.view(*original_shape)

        if flip_coord:
            pixel_coord = pixel_coord.flip(-1)

        return pixel_coord

    def pixel_coord_to_obs_coord_v2(self, coord) -> Tuple[float, float]:
        image_width = self.stats["image_width"]
        obs_min_total = self.stats["obs_min_total"]
        obs_range_total = self.stats["obs_range_total"]

        ii = image_width - coord[0]
        jj = coord[1]

        jj = jj / image_width * obs_range_total + obs_min_total
        ii = ii / image_width * obs_range_total + obs_min_total

        return (jj.item(), ii.item())

    def pixel_coord_to_obs_coord(self, coord) -> Tuple[float, float]:
        if "small" in self.env_name:
            return self.pixel_coord_to_obs_coord_v2(coord)

        if self.transformed:
            coord[0] = coord[0] * self.scale_factor + self.crop_left_top[0]
            coord[1] = coord[1] * self.scale_factor + self.crop_left_top[1]

        coord = (
            torch.as_tensor(coord, dtype=torch.float32) / self.scale_coord_obs_to_pixel
        )
        coord[1].neg_()
        x, y = coord + self.image_topleft_in_obs_coord

        return (x.item(), y.item())


def plot_obs_with_blur(obs, coord, save_name, reverse_input_coord=False):
    """
    A debugging function.
    obs: C x H x W torch tensor
    pixel coord: (x,y)

    it will produce a blur at location of (x, y) on the image. and save it
    """
    rounded_obs = coord.round().int().tolist()
    x, y = rounded_obs

    if reverse_input_coord:
        x, y = y, x

    copy_obs = obs.clone()

    # need to invert (x,y). dun worry about it
    copy_obs[:, x, y] = torch.tensor([227, 0, 0], dtype=torch.uint8)
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(copy_obs)

    image_pil.save(f"imgs/obs_with_red_dot_{save_name}.png")


def ij_to_obs(ij, obs_min_total, obs_range_total, n):
    """
    Calculate the observation coordinate (x, y) for a given (i, j) grid block.

    Parameters:
        ij (tuple): A tuple of integers (i, j) representing the grid block indices.
        obs_min_total (float): The minimum value in the observation space.
        obs_range_total (float): The total width/height of the observation space.
        n (int): The number of grid divisions along one dimension (n x n grid).

    Returns:
        numpy: A tensor of shape (..., 2) representing the observation coordinates (x, y).
    """
    # Calculate the size of each grid block
    grid_size = obs_range_total / n

    # Calculate the lower-left corner of the grid block
    lower_left = torch.tensor(ij, dtype=torch.float32) * grid_size + obs_min_total

    # Calculate the center of the grid block
    obs = lower_left + grid_size / 2

    return obs.numpy()


def ij_to_obs_random(grid_indices, obs_min_total, obs_range_total, n):
    """
    Calculate a random observation coordinate within a given (i, j) grid block,
    ensuring the point is away from the grid borders by a margin.

    Parameters:
        grid_indices (tuple): A tuple of integers (i, j) representing the grid block indices.
        obs_min_total (float): The minimum value in the observation space.
        obs_range_total (float): The total width/height of the observation space.
        n (int): The number of grid divisions along one dimension (n x n grid).

    Returns:
        numpy: A tensor of shape (..., 2) representing the random observation coordinates (x, y).
    """
    # Calculate the size of each grid block
    grid_size = obs_range_total / n
    margin = grid_size / 4  # Margin is 1/4 of the grid size

    # Calculate the lower-left corner of the grid block
    lower_left = (
        torch.tensor(grid_indices, dtype=torch.float32) * grid_size
        + obs_min_total
        + margin
    )

    # Calculate the range for the random sampling within the block
    range_size = grid_size - 2 * margin

    # Sample a random point within the block
    random_offset = torch.rand(2) * range_size
    obs = lower_left + random_offset

    return obs.numpy()


def obs_to_ij(obs, obs_min_total, obs_range_total, n):
    """
    Calculate the (i, j) grid block for a given observation.

    Parameters:
        obs (numpy): A tensor of shape (..., 2) representing the observation coordinates (x, y).
        obs_min_total (float): The minimum value in the observation space.
        obs_range_total (float): The total width/height of the observation space.
        n (int): The number of grid divisions along one dimension (n x n grid).

    Returns:
        tuple of integers (i, j) grid block indices.
    """
    # Calculate the size of each grid block
    grid_size = obs_range_total / n

    # Normalize the observation and calculate the indices
    grid_indices = (obs - obs_min_total) // grid_size

    return tuple(grid_indices.astype(int))


def sample_unique_a_tuple(tuples_list):
    """
    tuples_list: list of tuples. each tuple is of the form ((x,y), a, r)
    """
    # Count occurrences of each "a" value
    a_count = {}
    for _, a, _ in tuples_list:
        a_count[a] = a_count.get(a, 0) + 1

    # Filter tuples with unique "a" values
    unique_a_tuples = [t for t in tuples_list if a_count[t[1]] == 1]

    # Randomly select a tuple from the remaining tuples
    if unique_a_tuples:
        return random.choice(unique_a_tuples)
    else:
        return None  # No tuple meets the criteria


def sample_nearby_grid_location_ij(
    anchor,
    map_key,
    min_block_radius,
    max_block_radius,
    unique_shortest_path=False,
):
    map_layout = map_key.split("\\")

    # find block indices within max_block_radius
    neighbor_indices = find_reachable_positions_with_turns(
        map_layout,
        anchor[0],
        anchor[1],
        min_block_radius,
        max_block_radius,
    )

    # sample a pixel coordinate within any of those block indices
    if unique_shortest_path:
        sampled = sample_unique_a_tuple(neighbor_indices)
        if sampled is None:
            sampled_block_index, block_dist, turns = random.choice(neighbor_indices)
            unique_path = False
        else:
            sampled_block_index, block_dist, turns = sampled
            unique_path = True
    else:
        sampled_block_index, block_dist, turns = random.choice(neighbor_indices)
        unique_path = sum(1 for _, a, _ in neighbor_indices if a == block_dist) == 1

    return sampled_block_index, block_dist, turns, unique_path


def sample_nearby_grid_location_v2(
    anchor,
    map_key,
    min_block_radius,
    max_block_radius,
    obs_range_total,
    obs_min_total,
    unique_shortest_path,
):
    map_layout = map_key.split("\\")

    ij = obs_to_ij(anchor, obs_min_total, obs_range_total, n=len(map_layout))

    sampled_block_index, block_dist, turns, unique_path = (
        sample_nearby_grid_location_ij(
            ij,
            map_key,
            min_block_radius,
            max_block_radius,
            unique_shortest_path,
        )
    )

    obs = ij_to_obs_random(
        sampled_block_index, obs_min_total, obs_range_total, n=len(map_layout)
    )

    return obs, block_dist, turns, unique_path


def sample_nearby_grid_location(
    anchor,
    map_key,
    min_block_radius,
    max_block_radius,
    num_blocks,
    img_size,
    reverse_output_coord=False,
):
    """
    Function for sampling an (x,y) pixel coordinate from nearby grids
    Parameters:
        anchor: (x,y) pixel coordinate of origin
        map_key: the layout of the map in string format. exluding open surround
        max_block_radius: within how many blocks to sample from
        num_blocks: the number of blocks in image's width. including open surround
    """
    # figure out which block anchor corresponds to
    map_layout = map_key.split("\\")

    if num_blocks > len(map_layout):
        # it means there are surrounding space blocks. append them to layout
        assert num_blocks - len(map_layout) == 2
        border_row = "_" * num_blocks
        map_layout = (
            [border_row] + ["_" + row + "_" for row in map_layout] + [border_row]
        )

    def rotate_left_90(grid):
        # First, convert each row (string) into a list of characters for easier manipulation
        matrix = [list(row) for row in grid]
        rotated = list(zip(*matrix))
        rotated_grid = ["".join(row) for row in rotated][::-1]
        return rotated_grid

    map_layout = rotate_left_90(map_layout)

    anchor_block_pos = get_block_index(
        anchor[0],
        anchor[1],
        img_size=img_size,
        num_blocks=num_blocks,
        reverse_output_coord=True,
    )

    # find block indices within max_block_radius
    neighbor_indices = find_reachable_positions_with_turns(
        map_layout,
        anchor_block_pos[0],
        anchor_block_pos[1],
        min_block_radius,
        max_block_radius,
    )

    # sample a pixel coordinate within any of those block indices
    sampled_block_index, block_dist, turns = random.choice(neighbor_indices)

    sampled_pixel_coord = sample_image_coordinate_within_block(
        block_index=sampled_block_index,
        num_blocks=num_blocks,
        img_size=img_size,
    )

    if reverse_output_coord:
        sampled_pixel_coord = sampled_pixel_coord[::-1].copy()

    return sampled_pixel_coord, block_dist, turns


def get_block_index(x, y, img_size, num_blocks, reverse_output_coord=True):
    """
    Parameters:
        (x,y) pixel coordinate of the location
        num_blocks: number of grids/blocks in the environment width/height.
    Calculates the block index corresponding to the pixel coordinate
    """
    block_size = img_size / num_blocks

    i = int(y // block_size)
    j = int(x // block_size)

    if reverse_output_coord:
        return j, i
    else:
        return i, j


def find_reachable_positions_with_turns(map_, x, y, min_block_radius, max_block_radius):
    # Get the size of the map (n x n)
    n = len(map_)

    # Directions for moving up, down, left, right
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    direction_names = ["R", "L", "D", "U"]  # Names for easier tracking of turns

    # A dictionary to store the reachable positions, distances, and turn counts
    reachable = {}

    # Initialize the BFS queue with (current x, y, distance, turns, last direction)
    queue = deque([(x, y, 0, 0, None)])  # None as initial last_direction
    reachable[(x, y)] = (0, 0)  # Origin point has distance 0 and 0 turns

    # BFS to explore reachable positions within max_block_radius
    while queue:
        cur_x, cur_y, dist, turns, last_direction = queue.popleft()

        # Stop searching if we reach max_block_radius
        if dist >= max_block_radius:
            continue

        # Explore neighboring cells
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = cur_x + dx, cur_y + dy
            new_direction = direction_names[i]

            # Check if the new position is within bounds
            if 0 <= new_x < n and 0 <= new_y < n:
                # Check if the new position is 'O' and not yet visited
                if map_[new_x][new_y] == "O" and (new_x, new_y) not in reachable:
                    # Increment the turn count if there's a change in direction
                    new_turns = turns
                    if last_direction and new_direction != last_direction:
                        new_turns += 1

                    reachable[(new_x, new_y)] = (dist + 1, new_turns)
                    queue.append((new_x, new_y, dist + 1, new_turns, new_direction))

    # Convert the dictionary to a list of tuples (position, distance, turns)
    output = [
        ((pos[0], pos[1]), dist_turns[0], dist_turns[1])
        for pos, dist_turns in reachable.items()
    ]

    # Filter by min_block_radius
    filtered_output = [x for x in output if x[1] >= min_block_radius]

    if filtered_output:
        return filtered_output
    else:
        # If no block >= min_block_radius, return the max distance block
        return [output[-1]]


def sample_image_coordinate_within_block(
    block_index, num_blocks, img_size, padding=0.7
):
    i, j = block_index

    block_size = img_size / num_blocks  # Size of each block in the image

    # Calculate the pixel bounds for block (i, j) in the image
    x_min = i * block_size
    x_max = (i + 1) * block_size
    y_min = j * block_size
    y_max = (j + 1) * block_size

    # Sample a random coordinate within the block (i, j)
    sampled_x = random.uniform(x_min + padding, x_max - padding)
    sampled_y = random.uniform(y_min + padding, y_max - padding)

    return np.array([sampled_x, sampled_y])


def load_uniform(env_name, data_path):
    """
    Return:
        dict (map_idx: numpy(frames, obs_dim))
    """

    shape = None
    if "ant" in env_name.lower():
        shape = 29
    else:
        shape = 4

    uniform_obs_by_map = {}

    uniform_ds = torch.load(f"{data_path}/data.p")

    for i in range(len(uniform_ds)):
        episode = uniform_ds[i]

        if episode["map_idx"] in uniform_obs_by_map:
            uniform_obs_by_map[episode["map_idx"]].append(episode["observations"])
        else:
            uniform_obs_by_map[episode["map_idx"]] = [episode["observations"]]

    for map_idx in uniform_obs_by_map:
        uniform_obs_by_map[map_idx] = np.stack(uniform_obs_by_map[map_idx]).reshape(
            -1, shape
        )

    return uniform_obs_by_map
