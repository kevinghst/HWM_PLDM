import numpy as np


# Define the tapered distribution
def tapered_distribution(x, a, L, sigma, center):
    if np.abs(x - center) <= a:
        return 1.0  # Flat region, assign a constant value
    elif a < np.abs(x - center) <= L:
        return np.exp(
            -((np.abs(x - center) - a) ** 2) / (2 * sigma**2)
        )  # Gaussian tapering
    else:
        return 0.0  # Outside the range


# Vectorized function for sampling
def sample_tapered_distribution(a, L, sigma, center, size=1000):
    """
    a: Flat region boundary, increase for a wider flat region
    L: overall range, sets the range for the entire distribution
    sigma: Gaussian width
    center: Center of the distribution (middle of range)
    size: number of samples to generate
    """
    samples = np.random.uniform(center - L, center + L, size)
    pdf_values = np.array(
        [tapered_distribution(x, a, L, sigma, center) for x in samples]
    )

    # Normalize the PDF values and sample based on probability weights
    pdf_values /= pdf_values.sum()
    return np.random.choice(samples, size=size, p=pdf_values)
