from typing import Optional

import statsmodels.api as sm
import torch


def calc_avg_steps_to_goal(reward_history, reduce_type="mean"):
    reward_history_t = torch.stack(reward_history).transpose(0, 1)
    success_rows = torch.nonzero(reward_history_t.sum(dim=1) > 0).squeeze(-1).tolist()

    if len(success_rows) == 0:
        return -1
    steps_to_goal = []
    for i in success_rows:
        for j in range(len(reward_history_t[i])):
            if reward_history_t[i][j]:
                steps_to_goal.append(j)
                break

    if reduce_type == "mean":
        return torch.mean(torch.tensor(steps_to_goal, dtype=torch.float))
    elif reduce_type == "median":
        return torch.median(torch.tensor(steps_to_goal, dtype=torch.float))
    else:
        raise ValueError(f"Unknown reduce type: {reduce_type}")


def get_lr_p_results(features, outcomes):
    """
    Args:
        features: (bs,) dtype torch float32
        outcomes: (bs,) dtype torch float32 (0 or 1)
    Description:
        Use logistic regression for numeric features and binary outcomes to get correlation coefficient
    """
    outcomes = outcomes.cpu().numpy()
    features = features.cpu().numpy()
    features = sm.add_constant(features)
    model = sm.Logit(outcomes, features)
    result = model.fit()
    return result.params[1]


def normalize_actions(
    actions: torch.Tensor,
    min_angle: float = -1.74,
    max_angle: float = 1.71,
    min_norm: float = -2.27,
    max_norm: float = 2.27,
    eps: float = 1e-6,
    clamp_actions: bool = False,
    env_name: Optional[str] = None,
    dynamic_bounds: Optional[dict] = None,
):
    if clamp_actions:
        # we bound the absolute values instead of norm

        # L2 latent action bounds
        if (
            dynamic_bounds is not None
            and "min_bounds" in dynamic_bounds
            and "max_bounds" in dynamic_bounds
        ):
            abs_min_norm = dynamic_bounds["min_bounds"]
            abs_max_norm = dynamic_bounds["max_bounds"]

            min_norm = (abs_min_norm + 0.1).to(actions.device)
            max_norm = (abs_max_norm - 0.1).to(actions.device)

            actions_n = torch.clamp(
                actions, min=min_norm.view(1, -1), max=max_norm.view(1, -1)
            )

        elif actions.shape[-1] == 8:
            # TODO: FIX THIS HACK
            # L1 action bounds for ant environment
            abs_min_norm = torch.tensor(
                [-1.2022, -1.1061, -1.3260, -1.1111, -1.2677, -1.4228, -1.3263, -1.5168]
            )
            abs_max_norm = torch.tensor(
                [1.3606, 1.5552, 1.2483, 1.4850, 1.2797, 1.1922, 1.2396, 1.1205]
            )

            min_norm = (abs_min_norm + 0.1).to(actions.device)
            max_norm = (abs_max_norm - 0.1).to(actions.device)

            actions_n = torch.clamp(
                actions, min=min_norm.view(1, -1), max=max_norm.view(1, -1)
            )
        elif "cube" in env_name:
            abs_min_norm = torch.tensor([-2.0940, -1.6776, -1.2113, -1.6365, -1.1442])
            abs_max_norm = torch.tensor([2.0643, 1.6851, 1.2324, 1.6066, 1.2022])

            min_norm = (abs_min_norm + 0.1).to(actions.device)
            max_norm = (abs_max_norm - 0.1).to(actions.device)

            actions_n = torch.clamp(
                actions, min=min_norm.view(1, -1), max=max_norm.view(1, -1)
            )

        else:
            actions_n = torch.clamp(actions, min=min_norm, max=max_norm)
        return actions_n

    # we calculate norms of actions
    norms = actions.norm(dim=-1, keepdim=True)  # [300, 4, 1]
    # calculate min and max allowed step sizes
    max_norms = torch.ones_like(norms) * max_norm
    min_norms = torch.ones_like(norms) * min_norm

    # coeff is either 1 if the norm is below max, or max_norm / norm if it is above
    # coeff = torch.min(norms, max_norms) / (norms + eps)

    # coefficients for normalization
    coeff = torch.min(torch.max(norms, min_norms), max_norms) / (norms + eps)

    # rescale the actions
    return actions * coeff
