from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import torch
import argparse

from environments.diverse_maze import ant_draw
from torchvision import transforms
import imageio


def save_image(array, path):
    # Convert the numpy array to an image
    image = Image.fromarray(np.uint8(array))
    # Save the image in png format
    image.save(path, format="png")


def resize_image(images_np, image_transform):
    img = Image.fromarray(images_np)
    resized_img = image_transform(img)
    return np.array(resized_img)


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

    image_transform = transforms.Resize(tuple(config["img_size"][:2]))

    print(f"{len(all_splits)=}")
    if args.workers_num is None:
        start = 0
        end = len(all_splits)
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

    all_splits_current_worker = []

    env = None
    last_map_idx = None

    for split_idx in indices:
        split = all_splits[split_idx]
        map_idx = split["map_idx"]

        split_images = []

        if last_map_idx is not None and map_idx != last_map_idx:
            raise ValueError(
                "map changed!"
            )  # reinit env with different map may result in black images

        if env is None:
            print(f"Loading environment for map ID {map_idx}")
            env = ant_draw.load_environment(
                name=f"{config['env']}_{map_idx}",
                map_key=map_metadata[map_idx],
                max_episode_steps=config["episode_length"],
                no_legs=config["no_legs"],
            )
            last_map_idx = map_idx

        for img_idx, obs in tqdm(enumerate(split["observations"])):
            image_path = output_image_path / f"{split_idx}_{img_idx}.png"
            if os.path.exists(image_path) and not args.save_replace:
                continue

            options = {
                "reset_exact": True,
                "start_qpos": obs[:15],
                "start_qvel": obs[15:],
            }
            ob, _ = env.reset(options=options)
            visual_ob = env.render()

            if np.count_nonzero(visual_ob == 0):
                raise ValueError(f"Image contains zero values: {visual_ob}")

            visual_ob = resize_image(visual_ob, image_transform)
            split_images.append(visual_ob)

            if args.quick_debug and img_idx > 10:
                break

        split_images = np.stack(split_images)
        all_splits_current_worker.append(split_images)
        # imageio.mimsave(data_path / f"{split_idx}.gif", split_images, fps=15)

        if args.quick_debug and split_idx > 10:
            break

    torch.save(
        all_splits_current_worker,
        data_path / f"images_worker_{args.worker_id}_start_{start}_end_{end}.pt",
    )


if __name__ == "__main__":
    main()
