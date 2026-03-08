import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from hjepa.plotting.utils import paint_series

import environments.diverse_maze.ant_draw as ant_draw
import environments.diverse_maze.maze_draw as maze_draw
from environments.diverse_maze.utils import PixelMapper

# plan good:


# d4rl_medium_29

model_paths = {
    "gcbc": "/scratch/wz1232/ogbench/impls/exp/OGBench/Debug/ood-filtered-eval1-18-1-1-seed3_sd000_s_56652749.20250130_162910/all_trajs_step_1",
    "hiql": "/scratch/wz1232/ogbench/impls/exp/OGBench/Debug/ood-filtered-eval1-24-1-3_sd000_s_56652749.20250130_161931/all_trajs_step_1",
    "crl": "/scratch/wz1232/ogbench/impls/exp/OGBench/Debug/ood-filtered-eval1-27-1-seed3_sd001_s_56652749.20250130_162243/all_trajs_step_1",
    "gciql": "/scratch/wz1232/ogbench/impls/exp/OGBench/Debug/ood-filtered-eval1-15-2-1-seed3_sd000_s_56652749.20250130_162449/all_trajs_step_1",
    "hilp": "/scratch/wz1232/ogbench/impls/exp/OGBench/Debug/ood-filtered-eval1-21-1-6-seed3_sd000_s_56652749.20250130_162631/all_trajs_step_1",
    "pldm": [
        "/scratch/wz1232/HJEPA/checkpoint/maze2d_small_diverse/ood-filtered-full1-16-1-20-seed2/planning_l1_mpc_result_d4rl_medium",
        "/scratch/wz1232/HJEPA/checkpoint/maze2d_small_diverse/ood-filtered-full1-16-1-20-seed2/planning_l1_mpc_report_d4rl_medium",
    ],
}

models = ["pldm", "crl", "hiql", "gcbc", "gciql", "hilp"]
# models = ['pldm']


# color_codes = {
#     "gcbc": "#1f77b4",
#     "pldm": "#d62728",
#     "hiql": "#9467bd",
#     "crl": "#2ca02c",
#     "gciql": "#ff7f0e",
#     "hilp": "#8c564b",
#     # "hilp": "#ff7f0e",
#     # "gciql": "#8c564b",
# }

color_codes = {
    "gcbc": "#8c564b",
    "gciql": "#ff7f0e",
    "hilp": "#2ca02c",
    "pldm": "#d62728",
    "hiql": "#9467bd",
    "crl": "#1f77b4",
}

pldm_idx_map = {
    0: 0,
    16: 1,
    10: 2,
    12: 3,
    4: 4,
    6: 5,
    2: 6,
    8: 7,
    14: 8,
}

paint_start_pos = True

pixel_mapper = PixelMapper("maze2d_small_diverse")

trials = torch.load("/vast/wz1232/maze2d_small_diverse_5maps/id_ood_trials_filtered.pt")


for idx, _ in pldm_idx_map.items():

    map_layout = trials["map_layouts"][idx]
    start = trials["starts"][idx]
    target = trials["targets"][idx]
    ood = trials["ood_distance"][idx]

    env_name = "maze2d_small_diverse"
    env = ant_draw.load_environment(name=env_name, map_key=map_layout)
    drawer = maze_draw.create_drawer(env, env.name)

    # paint initial location

    # save image tensor
    # save PIL Image
    # image.save(f"/scratch/wz1232/HJEPA/icml_imgs/appendix/eval_map_layout_{idx}_start.png", format="PNG")
    # continue

    # paint trajs

    for model in models:

        render_location = start[:2]
        render_obs = np.pad(render_location, (0, 2), mode="constant")
        image = drawer.render_state(render_obs)
        image = Image.fromarray(image)
        center_crop = transforms.CenterCrop(346)
        image = center_crop(image)

        if model == "pldm":
            mpc_data = torch.load(model_paths[model][0])
            mpc_report = torch.load(model_paths[model][1])

            locations = [x[idx] for x in mpc_data.locations]
            terminate_idx = mpc_report.terminations[idx] + 10
            locations = locations[:terminate_idx]
        else:
            data = torch.load(model_paths[model])
            traj = data[idx + 1][0]
            locations = [x["location"] for x in traj["info"]]

        render_every = 1
        plot_locations = locations[::render_every]

        if "_" in model:
            base_model = model.split("_")[0]
        else:
            base_model = model

        color = color_codes[base_model]

        image = paint_series(
            image,
            pixel_mapper.obs_coord_to_pixel_coord_v2(
                torch.from_numpy(np.stack(plot_locations)), image_width=346
            ),
            dot_radius=2,
            color=color,
        )

        # paint target star
        # image = paint_at_xy_star(image, pixel_mapper.obs_coord_to_pixel_coord_v2(target, image_width=346), dot_radius=7, color='orange')

        center_crop = transforms.CenterCrop(260)
        image = center_crop(image)
        image.save(
            f"/scratch/wz1232/HJEPA/icml_imgs/appendix/trajs/{ood}_{model}_traj.png",
            format="PNG",
        )


# draw the map
