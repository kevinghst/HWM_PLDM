import concurrent
import collections
import os
from tqdm import tqdm
import logging
import argparse
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=20)
parser.add_argument("--env_name", type=str, required=True)

args = parser.parse_args()

log = logging.getLogger(__name__)


def list_images(metadata: dict):
    """This function should technically just list the images in the folder
    but it's slow so we're going to hardcode the list of images instead.

    The rest of the code doesn't assume anything about the number of
    episodes or that all episodes are of the same length.
    It only assumes that the images are named in the format
    "{episode_idx}_{timestep}.png".
    """

    image_list = []
    episode_length = metadata["episode_length"]
    n_episodes = metadata["n_episodes"]

    if "diverse" in args.env_name:
        n_episodes = n_episodes * metadata["train_maps_n"]

    for i in range(n_episodes):
        for j in range(episode_length + 1):
            image_list.append(f"{i}_{j}.png")

    return image_list


def remove_image(image_path):
    os.remove(image_path)


metadata_path = args.input_path + "/metadata.pt"
metadata = torch.load(metadata_path)
image_list = list_images(metadata=metadata)
num_images = len(image_list)

# Here we parse image file list into a dict mapping episode index to
# list of images along with timesteps.
# In general, when saving to zarr, we have to be careful
# to match the order of datapoints in the original non-image dataset.
# If the order doesn't match, the images will be misaligned with the
# rest of the data, and it's pretty hard to debug.

episode_dict = collections.OrderedDict()
for image_file in sorted(image_list):
    image_path = os.path.join(f"{args.input_path}/images", image_file)
    episode_idx, timestep = image_file.split("_")
    episode_idx = int(episode_idx)
    timestep = int(timestep[:-4])  # Strip ".png" extension
    if episode_idx not in episode_dict:
        episode_dict[episode_idx] = []
    episode_dict[episode_idx].append((timestep, image_path))

# Initialize the Zarr dataset

ctr = 0

log.info(f"Loading images from {args.input_path}/images")

# Loop over each episode and load the images
for i, (_episode_idx, image_list) in tqdm(
    enumerate(sorted(episode_dict.items())),
    desc="Loading images",
    total=len(episode_dict),
):
    # Sort images by timestep and load them into memory
    image_list = sorted(image_list, key=lambda x: x[0])
    images = []
    # for _j, (_, image_path) in enumerate(image_list):
    #     with Image.open(image_path) as image:
    #         image = image_transform(image)
    #         image = np.array(image)
    #         images.append(image)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        futures = []
        for _, image_path in image_list:
            future = executor.submit(remove_image, image_path)
            futures.append(future)

    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="Removing images",
    ):
        pass
