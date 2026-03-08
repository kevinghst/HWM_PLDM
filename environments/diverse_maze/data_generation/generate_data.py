import os
import sys
import argparse
import yaml
import torch


def main():
    parser = argparse.ArgumentParser(description="Generating Data for Mazes")
    parser.add_argument("--config", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--exclude_map_path", type=str, default=None)
    parser.add_argument("--gen_map_only", action="store_true")

    parser.add_argument("--map_start_idx", type=int, default=None)
    parser.add_argument("--map_end_idx", type=int, default=None)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    maps = torch.load(args.map_path) if args.map_path is not None else None
    exclude_maps = (
        torch.load(args.exclude_map_path) if args.exclude_map_path is not None else None
    )

    if (
        maps is not None
        and args.map_start_idx is not None
        and args.map_end_idx is not None
    ):
        filtered_maps = {}
        for key, value in maps.items():
            if args.map_start_idx <= int(key) <= args.map_end_idx:
                filtered_maps[key] = value
        maps = filtered_maps

    if "ant" in config["env"]:
        from environments.diverse_maze.data_generation.data_generator_antmaze import (
            AntMazeDataGenerator,
        )

        generator = AntMazeDataGenerator(
            config=config,
            output_path=args.output_path,
            maps=maps,
            exclude_maps=exclude_maps,
        )
    else:
        from environments.diverse_maze.data_generation.data_generator_maze2d import (
            PointMazeDataGenerator,
        )

        generator = PointMazeDataGenerator(
            config=config,
            output_path=args.output_path,
            maps=maps,
            exclude_maps=exclude_maps,
        )

    if not args.gen_map_only:
        generator.generate_data()


if __name__ == "__main__":
    main()
