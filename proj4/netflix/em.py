"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    total_log_likelihood = np.float64(0.0)
    for i in range(n):
        Cu = X[i].nonzero()[0]
        d_Cu = len(Cu)
        log_likelihood = np.float64(0.0)
        log_weighted_likelihoods = np.zeros(K)
        for j in range(K):
            # Log-Gaussian
            diff = X[i, Cu] - mixture.mu[j, Cu]
            square_distance = np.sum(diff ** 2)
            log_gaussian = square_distance/(-2. * mixture.var[j]) \
                - ((d_Cu / 2) * np.log(2 * np.pi * mixture.var[j]))
            # End Log-Gaussian
            log_weighted_likelihoods[j] = np.log(mixture.p[j]) + log_gaussian
        log_likelihood = logsumexp(log_weighted_likelihoods)
        post[i] = np.exp(log_weighted_likelihoods - log_likelihood)
        total_log_likelihood += log_likelihood

    return (post, total_log_likelihood)


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    return estep(X, mixture)


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
