# MAY NEED TO RUN THIS FILE FROM ROOT OF REPO TO WORK PROPERLY

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pldm_envs.diverse_maze import ant_draw
from pldm_envs.diverse_maze import maze_draw

offline_trials_path = "/scratch/wz1232/data/maze2d_large_diverse_probe/starts_targets_13_16.pt"
data_path = Path("/scratch/wz1232/data/maze2d_large_diverse_probe")

offline_trials = torch.load(offline_trials_path)

trajs = {
    # "HWM": "/scratch/wz1232/HJEPA/checkpoint/maze2d_large_diverse/l2_wo_encoder/l2_load_from_mine_249_seed246/planning_l2_mpc_result_l2_d4rl_hard",
    "PLDM": "/scratch/wz1232/HJEPA/checkpoint/maze2d_large_diverse/e2e_level1_mine_replicate_246_hard_eval/planning_l1_mpc_result_d4rl_hard"
}

config = torch.load(data_path / "metadata.pt")
map_metadata = torch.load(data_path / "train_maps.pt") if "diverse" in config["env"] else None
_drawer_cache = {}
center_crop = transforms.CenterCrop(370)

def get_drawer(map_idx=None):
    """Create the same drawer as render_data.py for this dataset."""
    if "diverse" in config["env"]:
        if map_idx is None:
            raise ValueError("map_idx is required for diverse environments")
        map_idx = int(map_idx)
        if map_idx not in _drawer_cache:
            env = ant_draw.load_environment(
                name=f"{config['env']}_{map_idx}",
                map_key=map_metadata[map_idx],
            )
            _drawer_cache[map_idx] = maze_draw.create_drawer(env, env.name)
        return _drawer_cache[map_idx]

    if "default" not in _drawer_cache:
        env = ant_draw.load_environment(config["env"])
        _drawer_cache["default"] = maze_draw.create_drawer(env, env.name)
    return _drawer_cache["default"]


def extract_goal_xy(trial_targets):
    target_arr = np.asarray(trial_targets)
    if target_arr.ndim == 1:
        return target_arr[:2]
    return target_arr[0, :2]


def center_crop_image(image):
    image_uint8 = np.asarray(image, dtype=np.uint8)
    image_pil = Image.fromarray(image_uint8)
    cropped = center_crop(image_pil)
    return np.asarray(cropped, dtype=np.uint8)

"""
data structure:

traj = {
    1: ...
    2: ...
}

traj[i][0] = {
    'observation': list of N
    'action':
    'reward':
    'done':
    'info': 
}



"""

for model_name, traj_path in trajs.items():
    traj = torch.load(traj_path)

    locations = traj.locations # list[max_num_steps] ([n_envs, 2]) torch.tensor
    targets = traj.targets # (n_envs, 2) torch.tensor
    reward_history = traj.reward_history # list[max_num_steps] (n_envs) torch.tensor

    def to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    locations = [to_numpy(step_locs) for step_locs in locations]
    targets = to_numpy(targets)
    reward_history = [to_numpy(step_rewards) for step_rewards in reward_history]

    targets_n_envs = int(targets.shape[0]) if targets.ndim > 0 else 0
    locations_n_envs = int(len(locations[0])) if locations else 0
    rewards_n_envs = int(len(reward_history[0])) if reward_history else 0
    n_envs = min(targets_n_envs, locations_n_envs, rewards_n_envs)
    max_steps = min(len(locations), len(reward_history))

    save_video_root = Path("videos") / model_name
    save_video_root.mkdir(parents=True, exist_ok=True)

    for episode_idx in range(n_envs):
        episode_map_idx = episode_idx % 20
        drawer = get_drawer(episode_map_idx)

        goal_xy = extract_goal_xy(targets[episode_idx])
        goal_obs = np.concatenate([goal_xy, np.zeros(2)])
        goal_image = center_crop_image(drawer.render_state(goal_obs))

        frames = []
        success = False
        for step_idx in range(max_steps):
            step_locations = locations[step_idx]
            step_rewards = reward_history[step_idx]
            if episode_idx >= len(step_locations) or episode_idx >= len(step_rewards):
                break

            curr_location = step_locations[episode_idx]
            curr_obs = np.concatenate([curr_location, np.zeros(2)])
            curr_image = center_crop_image(drawer.render_state(curr_obs))
            frames.append(curr_image)

            if step_rewards[episode_idx] == 1:
                success = True
                break

        save_video_path = save_video_root / f"episode_{episode_idx}_{success}.mp4"
        save_goal_path = save_video_root / f"episode_{episode_idx}_{success}_goal.png"

        imageio.imwrite(save_goal_path, goal_image)
        if frames:
            with imageio.get_writer(save_video_path, fps=20) as writer:
                for frame in frames:
                    writer.append_data(frame)
                    
