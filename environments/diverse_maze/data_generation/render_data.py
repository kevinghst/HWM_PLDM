from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import torch
import argparse

from environments.diverse_maze import ant_draw
from environments.diverse_maze import maze_draw


def save_image(array, path):
    # Convert the numpy array to an image
    image = Image.fromarray(np.uint8(array))
    # Save the image in png format
    image.save(path, format="png")


def main():
    parser = argparse.ArgumentParser(description="Convert proprio to images")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--workers_num", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--save_replace", action="store_true")
    parser.add_argument("--quick_debug", action="store_true")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_image_path = data_path / "images"
    output_image_path.mkdir(parents=True, exist_ok=True)
    proprio_path = data_path / "data.p"

    config_path = data_path / "metadata.pt"
    config = torch.load(config_path)

    # all_splits = d4rl_ds.splits
    all_splits = torch.load(proprio_path)

    print(f"{len(all_splits)=}")
    if args.workers_num is None:
        indices = np.arange(len(all_splits))
    else:
        per_worker = len(all_splits) // args.workers_num
        print(f"per_worker: {per_worker}")
        assert per_worker * args.workers_num == len(
            all_splits
        ), "Number of splits must be divisible by number of workers"
        start = per_worker * args.worker_id
        end = start + per_worker
        print(f"start: {start}, end: {end}")
        indices = np.arange(start, end)

    if "diverse" in config["env"]:
        # retrieve map metadata for generating custom environments
        map_metadata_path = data_path / "train_maps.pt"
        map_metadata = torch.load(map_metadata_path)
    else:
        env = ant_draw.load_environment(config["env"])
        drawer = maze_draw.create_drawer(env, env.name)

    for split_idx in indices:
        split = all_splits[split_idx]
        map_idx = split["map_idx"]

        if "diverse" in config["env"]:
            env = ant_draw.load_environment(
                name=f"{config['env']}_{map_idx}", map_key=map_metadata[map_idx]
            )
            drawer = maze_draw.create_drawer(env, env.name)

        for img_idx, obs in tqdm(enumerate(split["observations"])):
            image_path = output_image_path / f"{split_idx}_{img_idx}.png"
            if os.path.exists(image_path) and not args.save_replace:
                continue

            image = drawer.render_state(obs)
            save_image(image, image_path)

            if args.quick_debug and img_idx > 10:
                break

        # save one image per split per layout for visualization
        image_folder = data_path / f"layout_{map_idx}"
        if os.path.exists(image_folder):
            img_sample_path = image_folder / "image_sample.png"
            if not img_sample_path.exists() or args.save_replace:
                save_image(image, img_sample_path)

        if args.quick_debug and split_idx > 10:
            break


if __name__ == "__main__":
    main()
