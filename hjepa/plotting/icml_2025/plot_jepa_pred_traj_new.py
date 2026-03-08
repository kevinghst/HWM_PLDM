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


pixel_mapper = PixelMapper("maze2d_small_diverse")

mpc_data = torch.load(
    "/scratch/wz1232/HJEPA/checkpoint/maze2d_small_diverse/ood-filtered-full1-16-1-20-seed2/planning_l1_mpc_result_d4rl_medium"
)
trials = torch.load("/vast/wz1232/maze2d_small_diverse_5maps/id_ood_trials_filtered.pt")

mpc_report = torch.load(
    "/scratch/wz1232/HJEPA/checkpoint/maze2d_small_diverse/ood-filtered-full1-16-1-20-seed2/planning_l1_mpc_report_d4rl_medium"
)


# idxs = [29, 39, 36, 35, 32]

# selected_trials = {
#     'map_layouts': [],
#     'starts': [],
#     'targets': [],
# }
# for idx in idxs:
#     selected_trials['map_layouts'].append(trials['map_layouts'][idx])
#     selected_trials['starts'].append(trials['starts'][idx])
#     selected_trials['targets'].append(trials['targets'][idx])

# torch.save(selected_trials, '/vast/wz1232/maze2d_small_diverse/probe_train/start_target_planning_hard_selected.pt')


# time_idxs = [0, 5, 10, 15, 20]

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

idxs = list(pldm_idx_map.keys())
time_idx = 5

for idx in idxs:

    map_layout = trials["map_layouts"][idx]
    start = trials["starts"][idx]
    target = trials["targets"][idx]
    ood = trials["ood_distance"][idx]

    init_obs = mpc_data.observations[0][idx]

    preds = [x[:, idx] for x in mpc_data.pred_locations]
    locations = [x[idx] for x in mpc_data.locations]
    plot_every_n_pred = 1

    env_name = "maze2d_small_diverse"
    env = ant_draw.load_environment(name=env_name, map_key=map_layout)
    drawer = maze_draw.create_drawer(env, env.name)

    # draw the map

    curr_location = locations[time_idx]
    curr_preds = preds[time_idx][::plot_every_n_pred]
    curr_preds_xy = pixel_mapper.obs_coord_to_pixel_coord_v2(
        curr_preds, image_width=346
    )
    render_obs = np.pad(curr_location.numpy(), (0, 2), mode="constant")
    image = drawer.render_state(render_obs)

    image = Image.fromarray(image)

    center_crop = transforms.CenterCrop(346)

    image = center_crop(image)

    # paint predictions
    # curr_preds_xy = curr_preds_xy[:-10]
    painted_image = paint_series(
        image, curr_preds_xy, dot_radius=2, mark_dot=True, color="#FF00FF"
    )

    # paint target
    # painted_image = paint_at_xy_star(painted_image, pixel_mapper.obs_coord_to_pixel_coord_v2(target, image_width=346), dot_radius=7, color='orange')

    # painted_image = paint_at_xy(image, location_xy)

    center_crop = transforms.CenterCrop(260)
    painted_image = center_crop(painted_image)

    painted_image.save(f"icml_imgs/appendix/trajs/{ood}_pldm_plan.png", format="PNG")


# draw the map
