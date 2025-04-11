"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    weighted_likelihoods = np.zeros(K)

    log_likelihood = 0
    for i in range(n):
        likelihood = np.float64(0.0)
        weighted_likelihoods = np.zeros(K)
        for j in range(K):
            gaussian = np.exp(np.sum((X[i] - mixture.mu[j])**2)/(-2 * mixture.var[j])) / ((2 * np.pi * mixture.var[j])**(d/2))
            likelihood += np.sum(mixture.p[j] * gaussian)
            weighted_likelihoods[j] = mixture.p[j] * gaussian
        post[i] = weighted_likelihoods/likelihood
        log_likelihood += np.log(likelihood)
  
    return post, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    n_hat = np.sum(post, axis=0)
    p_hat = n_hat / n
    mu_hat = np.zeros((K,d))
    var_hat = np.zeros(K)

    for j in range(K):
        S1 = 0
        S2 = 0
        for i in range(n):
            S1 += post[i][j]*X[i]
        mu_hat[j] = S1 / n_hat[j]
        for i in range(n):
            S2 += post[i][j]*np.sum((X[i] - mu_hat[j])**2)
        var_hat[j] = S2 / (n_hat[j] * d)

    return GaussianMixture(mu_hat, var_hat, p_hat)


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
    old_log_likelihood = None
    log_likelihood = np.float64(0.0)
    while old_log_likelihood is None or np.abs(log_likelihood - old_log_likelihood) > 1e-6 * np.abs(log_likelihood):
      old_log_likelihood = log_likelihood
      post, log_likelihood = estep(X, mixture)
      mixture = mstep(X, post)

    return (mixture, post, log_likelihood)
