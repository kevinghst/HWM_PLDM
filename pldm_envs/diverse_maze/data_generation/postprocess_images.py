import concurrent
import collections
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from functools import partial
import sys
import argparse

import zarr
from pldm_envs.diverse_maze.transforms import select_transforms

import torch


BAD_EPISODES = {}


def process_image(transform, image_height, image_path: str):
    try:
        with Image.open(image_path) as image:
            image = transform(image)  # PIL.PngImagePlugin
            return np.array(image)
    except:
        episode_idx = int(image_path.split("/")[-1].split("_")[0])
        print(episode_idx)
        BAD_EPISODES[episode_idx] = True
        return np.zeros((image_height, image_height, 3))


def list_images(args, config: dict):
    """This function should technically just list the images in the folder
    but it's slow so we're going to hardcode the list of images instead.

    The rest of the code doesn't assume anything about the number of
    episodes or that all episodes are of the same length.
    It only assumes that the images are named in the format
    "{episode_idx}_{timestep}.png".
    """

    if args.skip_bad_episodes:
        bad_episodes = torch.load(f"{args.save_path}/bad_episodes.pt")
    else:
        bad_episodes = {}

    image_list = []
    episode_length = config["episode_length"]
    n_episodes = config["n_episodes"]

    if "diverse" in config["env"]:
        n_episodes = n_episodes * config["train_maps_n"]

    for i in range(n_episodes):
        if i in bad_episodes:
            continue

        if i >= args.stop_at_episode:
            break

        for j in range(episode_length + 1):
            image_list.append(f"{i}_{j}.png")

    return image_list


def main():
    parser = argparse.ArgumentParser(description="save images to numpy array")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--save_path", type=str, help="Path to save the numpy array")
    parser.add_argument("--quick_debug", action="store_true")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--stop_at_episode", type=int, default=sys.maxsize)
    parser.add_argument("--skip_bad_episodes", action="store_true")

    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.data_path

    config = torch.load(f"{args.data_path}/metadata.pt")

    image_list = list_images(args, config=config)
    num_images = len(image_list)

    # Here we parse image file list into a dict mapping episode index to
    # list of images along with timesteps.
    # In general, when saving to zarr, we have to be careful
    # to match the order of datapoints in the original non-image dataset.
    # If the order doesn't match, the images will be misaligned with the
    # rest of the data, and it's pretty hard to debug.

    input_image_path = os.path.join(args.data_path, "images")
    episode_dict = collections.OrderedDict()
    for image_file in sorted(image_list):
        image_path = os.path.join(input_image_path, image_file)
        episode_idx, timestep = image_file.split("_")
        episode_idx = int(episode_idx)
        timestep = int(timestep[:-4])  # Strip ".png" extension
        if episode_idx not in episode_dict:
            episode_dict[episode_idx] = []
        episode_dict[episode_idx].append((timestep, image_path))

    img_size = config["img_size"]
    h, w, ch = img_size
    data_shape = (num_images, *img_size)

    data_chunks = (10000, *img_size)

    # Initialize the Zarr dataset
    zarr_dataset = zarr.create(
        path=f"{args.data_path}/images.zarr",
        shape=data_shape,
        chunks=data_chunks,
        dtype=np.uint8,
    )

    image_transform = select_transforms(config["env"])
    process_image_with_transform = partial(process_image, image_transform, h)

    ctr = 0

    print(f"Loading images from {args.data_path}")

    # Loop over each episode and load the images
    for i, (_episode_idx, image_list) in tqdm(
        enumerate(sorted(episode_dict.items())),
        desc="Loading images",
        total=len(episode_dict),
        disable=not sys.stdout.isatty(),
    ):
        # Sort images by timestep and load them into memory
        image_list = sorted(image_list, key=lambda x: x[0])
        images = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            images = list(
                executor.map(
                    process_image_with_transform,
                    [image_path for _, image_path in image_list],
                )
            )

        zarr_dataset[ctr : ctr + len(images)] = np.stack(images)
        ctr += len(images)

        if args.quick_debug and i > 2:
            break

    if BAD_EPISODES:
        torch.save(BAD_EPISODES, f"{args.save_path}/bad_episodes.pt")
    else:
        np.save(f"{args.save_path}/images.npy", zarr_dataset)


if __name__ == "__main__":
    main()
