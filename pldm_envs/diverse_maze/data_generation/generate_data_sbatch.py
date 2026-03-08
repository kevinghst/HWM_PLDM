import os
import sys
import argparse
import yaml


import time
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Generating Data for Mazes")
    parser.add_argument("--config", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--map_path", type=str, default=None)
    parser.add_argument("--exclude_map_path", type=str, default=None)

    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--bash_script", type=str, default="run_4.sbatch")

    parser.add_argument("--render_only", action="store_true")

    args = parser.parse_args()

    if args.render_only:
        for i in range(args.n_gpus):
            command_str = " ".join(
                [
                    "python /scratch/wz1232/HJEPA/environments/diverse_maze/data_generation/render_and_process_data.py",
                    f"--data_path {args.output_path}",
                    f"--workers_num {args.n_gpus}",
                    f"--worker_id {i}",
                ]
            )

            bashCmd = [
                "sbatch",
                f"/scratch/wz1232/HJEPA/scripts/{args.bash_script}",
            ] + [command_str]

            print(args.bash_script)
            print(command_str)
            print("\n")
            process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
            output, error = process.communicate()
            time.sleep(1)

    else:
        import torch

        assert args.map_path is not None
        maps = torch.load(args.map_path) if args.map_path is not None else None

        map_keys = list(maps.keys())

        # break it into chunks of n_gpus
        num_maps = len(map_keys)
        chunk_size = num_maps // args.n_gpus
        chunks = [map_keys[i : i + chunk_size] for i in range(0, num_maps, chunk_size)]

        for chunk in chunks:
            command_str = " ".join(
                [
                    "python /scratch/wz1232/HJEPA/environments/diverse_maze/data_generation/generate_data.py",
                    f"--config {args.config}",
                    f"--output_path {args.output_path}/maps_{chunk[0]}_{chunk[-1]}",
                    f"--map_path {args.map_path}",
                    f"--map_start_idx {chunk[0]}",
                    f"--map_end_idx {chunk[-1]}",
                ]
            )

            if args.exclude_map_path is not None:
                command_str += f" --exclude_map_path {args.exclude_map_path}"

            bashCmd = [
                "sbatch",
                f"/scratch/wz1232/HJEPA/scripts/{args.bash_script}",
            ] + [command_str]

            print(args.bash_script)
            print(command_str)
            print("\n")
            process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
            output, error = process.communicate()
            time.sleep(1)


if __name__ == "__main__":
    main()
