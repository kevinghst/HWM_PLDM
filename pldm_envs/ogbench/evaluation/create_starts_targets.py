import yaml
import torch
from pldm_envs.ogbench.dataset import LocoMazeDataset
import dataclasses
from pldm.utils import update_config_from_yaml
from pldm.data.utils import make_dataloader

from pldm_envs.diverse_maze.enums import D4RLDatasetConfig

data_yaml_path = "/scratch/wz1232/HJEPA/hjepa/configs/visual_ant/medium_explore.yaml"
offline_data_path = (
    "/vast/wz1232/ogbench/visual_ant_maze_medium_stitch_1M_noise0.2/val_50000"
)
sample_length = 100
num_trajs = 100

with open(data_yaml_path, "r") as file:
    yaml_data = yaml.safe_load(file)

data_config = update_config_from_yaml(
    D4RLDatasetConfig, yaml_data["data"]["d4rl_config"]
)

ds = LocoMazeDataset(
    dataclasses.replace(
        data_config,
        batch_size=num_trajs,
        normalize=False,
        path=offline_data_path,
        load_top_down_view=True,
        num_workers=1,
        sample_length=sample_length,  # some wiggle room
    )
)

ds = make_dataloader(ds=ds)

datum = next(iter(ds))

(
    states,
    locations,
    actions,
    indices,
    proprio_pos,
    proprio_vel,
    top_down_view_states,
) = datum

print(states.shape)

proprio_pos_w_loc = torch.cat([locations.squeeze(-2), proprio_pos], dim=-1)

save_dict = {
    "states": states,
    "actions": actions,
    "indices": indices,
    "proprio_pos": proprio_pos_w_loc,
    "proprio_vel": proprio_vel,
    "top_down_view_states": top_down_view_states,
}

torch.save(save_dict, f"{offline_data_path}/plan_start_target_len_{sample_length}.pt")
