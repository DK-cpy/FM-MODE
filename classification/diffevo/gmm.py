import torch
from sklearn.mixture import GaussianMixture
import numpy as np

def distance_matrix(x, y):
    """Compute the pairwise distance matrix between x and y.

    Args:
        x: (N, d) tensor.
        y: (M, d) tensor.
    Returns:
        (N, M) tensor, the pairwise distance matrix.
    """
    return torch.cdist(x, y)


def GMM(samples, num_components=5):
    """Density estimation using Gaussian Mixture Model (GMM).

    Args:
        samples: (N, d) tensor, the samples to estimate the density.
        num_components: int, the number of Gaussian components to use.
    Returns:
        (N,) tensor, the estimated density at the given samples.
    """
    # Convert samples to numpy array for sklearn compatibility
    samples_np = samples.numpy()

    # Fit the GMM model
    gmm = GaussianMixture(n_components=num_components)
    gmm.fit(samples_np)

    # Compute the log likelihood of each sample
    log_likelihood = gmm.score_samples(samples_np)

    # Convert log likelihood to density
    density_estimates = torch.from_numpy(np.exp(log_likelihood))

    return density_estimates