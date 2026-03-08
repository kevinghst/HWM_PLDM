import numpy as np
from environments.wall.wall import DotWall
from environments.wall.wrappers import NormEvalWrapper
import torch


def construct_eval_envs(
    seed,
    wall_config,
    n_envs: int,
    level: str,
    cross_wall: bool = True,
    normalizer=None,
    trials_path=None,
):
    rng = np.random.default_rng(seed)

    if trials_path is not None:
        trials = torch.load(trials_path)
    else:
        trials = None

    envs = [
        DotWall(
            rng=rng,
            border_wall_loc=wall_config.border_wall_loc,
            wall_width=wall_config.wall_width,
            door_space=wall_config.door_space,
            wall_padding=wall_config.wall_padding,
            img_size=wall_config.img_size,
            fix_wall=wall_config.fix_wall,
            cross_wall=cross_wall,
            level=level,
            n_steps=wall_config.n_steps,
            action_step_mean=wall_config.action_step_mean,
            max_step_norm=wall_config.action_upper_bd,
            fix_wall_location=wall_config.fix_wall_location,
            fix_door_location=wall_config.fix_door_location,
        )
        for _ in range(n_envs)
    ]

    for i, e in enumerate(envs):
        if trials is not None:
            e.reset(
                dot_position=trials["dot_positions"][i],
                target_position=trials["target_positions"][i],
            )
        else:
            e.reset()

    if normalizer is not None:
        envs = [NormEvalWrapper(e, normalizer=normalizer) for e in envs]

    # dump dot_position and target_position
    # attr = {
    #     "dot_positions": [],
    #     "target_positions": []
    # }

    # for e in envs:
    #     attr['dot_positions'].append(e.dot_position.cpu())
    #     attr["target_positions"].append(e.target_position.cpu())

    # torch.save(attr, '/vast/wz1232/wall/25_trials.pt')

    return envs
