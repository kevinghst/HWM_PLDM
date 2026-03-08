import torch
import networkx as nx
import numpy as np

from plotting.utils import load_uniform_umaze, load_expert_medium, load_uniform_large


def select_subset(x, D=2000):
    indices = np.random.choice(x.shape[0], D, replace=False)
    return x[indices]


def generate_grid(uniform_obs, N=11, d_max=0.005):
    min_x = uniform_obs[:, 0].min()
    max_x = uniform_obs[:, 0].max()
    min_y = uniform_obs[:, 1].min()
    max_y = uniform_obs[:, 1].max()

    buffer = 0.2
    xs = torch.linspace(min_x + buffer, max_x - buffer, N)
    ys = torch.linspace(min_y + buffer, max_y - buffer, N)

    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    xx = xx.flatten()
    yy = yy.flatten()

    obs_subset = torch.from_numpy(select_subset(uniform_obs, 5000))

    diffs_x = xx.unsqueeze(-1) - obs_subset[:, 0].unsqueeze(0)
    diffs_y = yy.unsqueeze(-1) - obs_subset[:, 1].unsqueeze(0)

    distances = diffs_x.pow(2) + diffs_y.pow(2)
    min_dists = distances.min(dim=-1).values

    mask = min_dists < d_max

    xx_f = xx[mask]
    yy_f = yy[mask]

    starts = torch.stack([xx_f, yy_f], dim=-1)
    return starts


def create_graph(V, d_max):
    pairwise_distances = torch.cdist(V, V)
    A = torch.where(
        pairwise_distances <= d_max,
        torch.ones_like(pairwise_distances),
        torch.zeros_like(pairwise_distances),
    )
    return A


def create_nx_graph(V, A):
    G = nx.from_numpy_array(A.astype(np.int32))
    mapping = {i: tuple([round(x, 1) for x in coord]) for i, coord in enumerate(V)}
    nx.set_node_attributes(G, mapping, "pos")
    return G


def visualize_nx_graph(G):
    pos = nx.get_node_attributes(G, "pos")
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="skyblue",
        edge_color="gray",
        font_size=4,
    )


def closest_point(c, V):
    diff = V - c
    dists = np.linalg.norm(diff, axis=1)
    return np.argmin(dists)


UMAZE_GRID_N = 11
UMAZE_GRID_DMAX = 0.005
UMAZE_GRAPH_DMAX = 0.8
UMAZE_GOAL_HORIZON = 3

MEDIUM_GRID_N = 17
MEDIUM_GRID_DMAX = 0.02
MEDIUM_GRAPH_DMAX = 1
MEDIUM_GOAL_HORIZON = 3

LARGE_GRID_N = 17
LARGE_GRID_DMAX = 0.02
LARGE_GRAPH_DMAX = 1
LARGE_GOAL_HORIZON = 3


class ExpertPlanner:
    def __init__(self, env_name: str = "maze2d-umaze-v1"):
        if "umaze" in env_name:
            self.uniform_obs = load_uniform_umaze()
            self.N = UMAZE_GRID_N
            self.grid_dmax = UMAZE_GRID_DMAX
            self.d_max = UMAZE_GRAPH_DMAX
        elif "medium" in env_name:
            self.uniform_obs = load_expert_medium()
            self.N = MEDIUM_GRID_N
            self.grid_dmax = MEDIUM_GRID_DMAX
            self.d_max = MEDIUM_GRAPH_DMAX
        elif "large" in env_name:
            self.uniform_obs = load_uniform_large()
            self.N = LARGE_GRID_N
            self.grid_dmax = LARGE_GRID_DMAX
            self.d_max = LARGE_GRAPH_DMAX

        self.V = generate_grid(self.uniform_obs, N=self.N, d_max=self.grid_dmax)
        A = create_graph(self.V, self.d_max).numpy()
        self.V = self.V.numpy()
        self.G = create_nx_graph(self.V, A)

    def plan(self, start, goal):
        start_idx = closest_point(start, self.V)
        goal_idx = closest_point(goal, self.V)
        path = nx.astar_path(self.G, start_idx, goal_idx)
        path = np.array(path)
        return self.V[path]

    def plan_next_subgoal(self, start, goal):
        if np.linalg.norm(start - goal) < self.d_max:
            return goal
        plan = self.plan(start, goal)
        if len(plan) >= 3:
            return plan[2]
        else:
            return plan[1]

    def render_graph(self):
        visualize_nx_graph(self.G)
