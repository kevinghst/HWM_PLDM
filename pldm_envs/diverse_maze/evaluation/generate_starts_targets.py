"""
Generate a dictionary with keys values:
- starts: lists of (4,) numpy arrays
- targets: lists of (4,) numpy arrays
- map_layouts: list of strings of example '######\\##O###\\#OOOO#\\#OO#O#\\#OO#O#\\######'
- block_dists: list of integers
"""

import argparse
from pathlib import Path

import torch
from pldm_envs.diverse_maze.evaluation.maze2d_envs_generator import (
    Maze2DEnvsGenerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate starts/targets trial files for diverse maze evaluation"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to the probe dataset folder where starts_targets_*.pt files will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    block_radii = [(5, 8), (9, 12), (13, 16)]  # ez
    n_envs_per_setting = 40
    env_name = "maze2d_large_diverse"

    for min_r, max_r in block_radii:

        config = {
            "env_name": env_name,
            "n_envs": n_envs_per_setting,
            "min_block_radius": min_r,
            "max_block_radius": max_r,
            "action_repeat": 4,
            "action_repeat_mode": "id",
            "stack_states": 1,
            "image_obs": True,
            "data_path": str(data_path),
            "set_start_target_path": None,
            "unique_shortest_path": False,
        }

        # convert to object with attributes
        from types import SimpleNamespace

        config = SimpleNamespace(**config)

        envs_generator = Maze2DEnvsGenerator(
            env_name=config.env_name,
            n_envs=config.n_envs,
            min_block_radius=config.min_block_radius,
            max_block_radius=config.max_block_radius,
            action_repeat=config.action_repeat,
            action_repeat_mode=config.action_repeat_mode,
            stack_states=config.stack_states,
            image_obs=config.image_obs,
            data_path=config.data_path,
            trials_path=config.set_start_target_path,
            unique_shortest_path=config.unique_shortest_path,
            normalizer=None,
        )

        envs, trials = envs_generator()
        torch.save(trials, data_path / f"starts_targets_{min_r}_{max_r}.pt")

if __name__ == "__main__":
    main()
