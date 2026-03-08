import numpy as np


def sample_vector(min_norm=0, max_norm=1):
    # in reality d4rl point action norm can be 1.4 maximum norm. but we bound it at 1
    magnitude = np.random.uniform(min_norm, max_norm)
    angle = np.random.uniform(0, 2 * np.pi)
    x = magnitude * np.cos(angle)
    y = magnitude * np.sin(angle)
    return np.array([x, y])
