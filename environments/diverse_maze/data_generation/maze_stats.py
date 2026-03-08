STATS = {
    "maze2d-umaze-v1": {
        "n_episodes": 1000,
        "episode_length": 1000,
        "img_size": (64, 64, 3),
        "qvel_norm_prior": False,
    },
    "maze2d-medium-v1": {
        # "n_episodes": 2000,
        "n_episodes": 5,
        "episode_length": 1000,
        "img_size": (64, 64, 3),
        "qvel_norm_prior": False,
    },
    "maze2d-large-v1": {
        "n_episodes": 4000,
        "episode_length": 1000,
        "img_size": (64, 64, 3),
        "qvel_norm_prior": False,
    },
    "maze2d_medium_diverse": {
        "env": "maze2d_medium_diverse",
        "num_blocks_width_in_img": 10,  # including outer space surround
        "n_episodes": 250,
        "train_maps_n": 160,
        # for probing
        # "n_episodes": 100,
        # "train_maps_n": 40,
        "episode_length": 100,
        "img_size": (81, 81, 3),
        "qvel_norm_prior": True,
        "qvel_norm_prior_type": "uniform",  # ['normal', 'uniform']
        "action_repeat": 4,
        "action_repeat_mode": "id",  # ['linear', 'null']
        "resample_every": 1,
        "sampling_mode": "uniform",  # ['uniform', 'ou']
        "max_path_len": 12,
        "sparsity_low": 50,
        "sparsity_high": 80,
        # for ogbench gather function
        "ogbench_gather": False,
        "dataset_type": "explore",
        "sample_every": 1,
        "noise": 0,
    },
    "maze2d_small_diverse": {
        "env": "maze2d_small_diverse",
        "num_blocks_width_in_img": 8,  # including outer space surround
        # "n_episodes": 2000,
        # "train_maps_n": 5,
        "n_episodes": 250,
        "train_maps_n": 40,
        "episode_length": 100,
        "img_size": (64, 64, 3),
        "qvel_norm_prior": True,
        "qvel_norm_prior_type": "uniform",  # ['normal', 'uniform']
        "action_repeat": 4,
        "action_repeat_mode": "id",  # ['linear', 'null', 'id]
        "resample_every": 1,
        "action_noise": 0,
        "sampling_mode": "uniform",  # ['uniform', 'ou']
        "max_path_len": 8,
        "sparsity_low": 50,
        "sparsity_high": 75,
        "wall_coords": [(0, 0), (0, 1), (1, 0), (1, 1)],
        # "space_coords": [(0,0)],
    },
    "maze2d_small_diverse_single": {
        "env": "maze2d_small_diverse_single",
        "num_blocks_width_in_img": 8,  # including outer space surround
        "n_episodes": 10000,
        "train_maps_n": 1,
        # "n_episodes": 100,
        # "train_maps_n": 40,
        "episode_length": 100,
        "img_size": (64, 64, 3),
        "qvel_norm_prior": True,
        "qvel_norm_prior_type": "uniform",  # ['normal', 'uniform']
        "action_repeat": 4,
        "action_repeat_mode": "id",  # ['linear', 'null', 'id]
        "resample_every": 1,
        "action_noise": 0,
        "sampling_mode": "uniform",  # ['uniform', 'ou']
    },
}

RENDER_STATS = {
    "maze2d-umaze-v1": {
        "lookat": [3, 3, 0],
        "image_topleft_in_obs_coord": [-2.1729, 5.7517],
        "scale_coord_obs_to_pixel": 63,
        "arrow_mult": 30,
    },
    "maze2d-medium-v1": {
        "lookat": [4.5, 4.5, 0],
        "image_topleft_in_obs_coord": [-1.95, 8.55],
        "scale_coord_obs_to_pixel": (500 / 10.5),
        "arrow_mult": 25,
        "transformed": True,
        "crop_left_top": [100, 100],
        "scale_factor": 300 / 64,
    },
    # "maze2d_medium_diverse": {
    #     "lookat": [4.5, 4.5, 0],
    #     "image_topleft_in_obs_coord": [-1.95, 8.55],
    #     "scale_coord_obs_to_pixel": (500 / 10.5),
    #     "arrow_mult": 25,
    #     "transformed": True,
    #     "crop_left_top": [10, 10],
    #     "scale_factor": 6,
    # },
    "maze2d_large_diverse": {
        "lookat": [5.5, 5.5, 0],
        "image_topleft_in_obs_coord": [-1.95, 8.55],  # TO FIX
        "scale_coord_obs_to_pixel": (500 / 10.5),  # TO FIX
        "arrow_mult": 25,  # CHECK IF THIS IS ACTUALLY USED
        "transformed": True,
        "crop_left_top": [10, 10],  # deprecated
        "scale_factor": 386 / 98,  # deprecated
        "image_width": 98,
        "obs_min_space": 0.351,
        "obs_max_space": 8.248,
        # "obs_range_total": 9.87125,  # (obs_max_space - obs_min_space) / empty_blocks * total_blocks
        "obs_range_total": 10.068675,  # total_blocks = 10.2
        "obs_min_total": -0.636125,  # obs_min_space - obs_range_total / total_blocks
    },
    "maze2d_medium_diverse": {
        "lookat": [4.5, 4.5, 0],
        "image_topleft_in_obs_coord": [-1.95, 8.55],  # TO FIX
        "scale_coord_obs_to_pixel": (500 / 10.5),  # TO FIX
        "arrow_mult": 25,  # CHECK IF THIS IS ACTUALLY USED
        "transformed": True,
        "crop_left_top": [10, 10],
        "scale_factor": 386 / 81,
        "image_width": 81,
        "obs_min_space": 0.353,
        "obs_max_space": 6.246,
        "obs_range_total": 7.857,  # (obs_max_space - obs_min_space) / empty_blocks * total_blocks
        "obs_min_total": -0.629,  # obs_min_space - obs_range_total / total_blocks
    },
    "maze2d_small_diverse": {
        "lookat": [3.5, 3.5, 0],
        "image_topleft_in_obs_coord": [-1.95, 8.55],  # TO FIX
        "scale_coord_obs_to_pixel": (500 / 10.5),  # TO FIX
        "arrow_mult": 25,  # CHECK IF THIS IS ACTUALLY USED
        "transformed": True,
        "crop_left_top": [77, 77],
        "scale_factor": 345 / 64,
        "image_width": 64,
        "obs_min_space": 0.35,
        "obs_max_space": 4.2485,
        "obs_range_total": 5.84775,  # (obs_max_space - obs_min_space) / empty_blocks * total_blocks
        "obs_min_total": -0.6246,  # obs_min_space - obs_range_total / total_blocks
    },
    "maze2d-large-v1": {
        "lookat": [5.0, 6.5, 0],
        "image_topleft_in_obs_coord": [-3.06, 12.15],
        "scale_coord_obs_to_pixel": 36.5,
        "arrow_mult": 20,
    },
}
