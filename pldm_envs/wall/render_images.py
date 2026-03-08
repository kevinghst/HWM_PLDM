import argparse
import yaml

import numpy as np
import torch
from tqdm import tqdm

from pldm_envs.wall.data.wall import WallDataset, WallDatasetConfig
from pldm_envs.wall.save_wall_ds import update_config_from_yaml


def parse_args():
    """A function to parse arguments with argparse:
    - data_paths: a list of paths to the data files
    - wc_rate: target wall crossing rate in the new dataset
    - output_path: path to save the new dataset
    """

    parser = argparse.ArgumentParser(
        description="Render images in a dataset without image observations",
    )
    parser.add_argument("--input_path", type=str, help="Path to the data file")
    parser.add_argument("--config", type=str, help="Path to the dataset config file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="new_dataset.npz",
        help="Path to save the new dataset with images rendered.",
    )
    parser.add_argument(
        "--render_batch_size",
        type=int,
        default=1000,
        help="Number of trajectories render at a time",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input_path, mmap_mode="r")

    data = dict(data)

    with open(args.config, "r") as file:
        yaml_config = yaml.safe_load(file)

    config = update_config_from_yaml(WallDatasetConfig, yaml_config)
    ds = WallDataset(config)

    locations = data["locations"]

    images = []
    # we render each trajectory separately
    for i in tqdm(range(0, len(locations), args.render_batch_size)):
        sl = slice(i, min(i + args.render_batch_size, len(locations)))
        traj_slice = locations[sl]
        images.append(ds.render_location(torch.from_numpy(traj_slice)))

    images = torch.cat(images, dim=0)

    wall_info = ds.sample_walls()
    walls = ds.render_walls(*wall_info)
    # take the first wall
    walls = walls[0].unsqueeze(-1).repeat(images.shape[0], 1, 1, 1)

    images = images.unsqueeze(-1)
    images = torch.cat([images, walls], dim=-1)

    data["observations"] = images.numpy()

    np.savez(args.output_path, **data)


if __name__ == "__main__":
    main()
