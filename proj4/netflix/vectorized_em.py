"""Mixture model for matrix completion using matrix operations

The naive implementations work for complete cases
For incomplete cases, check estep and mstep
I added a compact version of each implementation.
Compact and not compact are the same implmentation just using less intermediate variables
"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture

def naive_estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    _, d = X.shape
    mu, var, p = mixture

    diff = X[:,np.newaxis,:] - mu[np.newaxis,:,:]
    square_distance = np.sum(diff**2, axis=2)
    power = square_distance / (-2 * var)
    gaussian = np.exp(power) / (2 * np.pi * var)**(d/2)
    weighted = gaussian * p[np.newaxis,:]
    likelihood = np.sum(weighted, axis=1, keepdims=True)
    post = weighted / likelihood
    log_likelihood = np.sum(np.log(likelihood))

    return post, log_likelihood

def naive_estep_compact(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    _, d = X.shape 
    mu, var, p = mixture

    weighted = (np.exp(np.sum((X[:,np.newaxis,:] - mu[np.newaxis,:,:])**2, axis=2) / (-2 * var)) / (2 * np.pi * var)**(d/2)) * p[np.newaxis,:]
    likelihood = np.sum(weighted, axis=1, keepdims=True)
    post = weighted / likelihood
    log_likelihood = np.sum(np.log(likelihood))

    return post, log_likelihood


def naive_mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
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

    mu_hat = post.T @ X / n_hat[:,np.newaxis]
    diff = X[:,np.newaxis,:] - mu_hat[np.newaxis,:,:]
    square_distance = np.sum(diff**2, axis=2)
    var_hat = np.sum(post * square_distance, axis=0) / (n_hat * d)

    return GaussianMixture(mu_hat, var_hat, p_hat)

def naive_mstep_compact(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    n, d = X.shape
    _, K = post.shape

    nh = np.sum(post, axis=0)
    ph = nh / n
    muh = post.T @ X / nh[:,np.newaxis]
    vah = np.sum(post * np.sum((X[:,np.newaxis,:] - muh[np.newaxis,:,:])**2, axis=2), axis=0) / (nh * d)

    return GaussianMixture(muh, vah, ph)


def naive_run(X: np.ndarray, mixture: GaussianMixture,
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
      post, log_likelihood = naive_estep_compact(X, mixture)
      mixture = naive_mstep_compact(X, post)

    return (mixture, post, log_likelihood)


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
    mu, var, p = mixture
    K, _ = mu.shape
    # Reshaping
    Xr = X[:,np.newaxis,:]     # (n, 1, d)
    mur = mu[np.newaxis,:,:]   # (1, K, d)
    varr = var[np.newaxis, :]  # (1, K)
    pr = p[np.newaxis, :]      # (1, K)
    # Cu is a mask of the observed values per point x cluster
    mask = (X != 0)                            # (n, d)
    mask_res = mask[:, np.newaxis, :]          # (n, 1, d)
    Cu = np.broadcast_to(mask_res, (n, K, d))  # (n, K, d)
    # d_Cu = number of observed dimensions per point x cluster
    d_Cu = np.sum(Cu, axis=2)                  # (n, K)

    sq_diff = np.sum(((Xr - mur)**2) * Cu, axis=2)        # (n, K, d) -> (n, K)
    component1 = (sq_diff / (-2 * varr))                   # (n, K)
    component2 = (-d_Cu / 2) * np.log(2 * np.pi * varr)   # (n, K) * (K) -> (n, K)
    log_gaussian = component1 + component2                # (n, K)
    log_weighted = np.log(pr) + log_gaussian              # (K) + (n, K) -> (n, K)
    log_likelihood = logsumexp(log_weighted, axis=1, keepdims=True)  # (n, 1)
    post = np.exp(log_weighted - log_likelihood)          # (n, K)
    total_log_likelihood = np.sum(log_likelihood)         # Scalar

    return (post, total_log_likelihood)

def estep_compact(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    mu, var, p = mixture
    K, _ = mu.shape
    varr = var[np.newaxis, :]  # Reshaping var (1, K)

    Cu = np.broadcast_to((X != 0)[:, np.newaxis, :], (n, K, d))  # (n, K, d)
    d_Cu = np.sum(Cu, axis=2)  # (n, K)

    log_gaussian = (np.sum(((X[:,np.newaxis,:] - mu[np.newaxis,:,:])**2) * Cu, axis=2) / (-2 * varr)) \
      + (-d_Cu / 2) * np.log(2 * np.pi * varr)                       # (n, K)
    log_weighted = np.log(p[np.newaxis, :]) + log_gaussian           # (n, K)
    log_likelihood = logsumexp(log_weighted, axis=1, keepdims=True)  # (n, 1)

    return (np.exp(log_weighted - log_likelihood), np.sum(log_likelihood))


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
    n, d = X.shape
    mu, _, _ = mixture
    K, _ = mu.shape
    n_hat = np.sum(post, axis=0)   # (K)
    p_hat = n_hat / n              # (K)

    # Cu is a mask of the observed values per point x cluster
    mask = (X != 0)                            # (n, d)
    mask_res = mask[:, np.newaxis, :]          # (n, 1, d)
    Cu = np.broadcast_to(mask_res, (n, K, d))  # (n, K, d)
    d_Cu = np.sum(mask, axis=1)                # (n)

    numerator = post.T @ X
    denominator = (post.T @ mask)  # (K, d)
    mu_raw = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator >= 1))
    mu_hat = np.where(denominator >= 1, mu_raw, mu)  # (K, d)

    sq_diff = np.sum(((X[:, np.newaxis, :] - mu_hat[np.newaxis, :, :])**2) * Cu, axis=2)  # (n, K)
    numerator = np.sum((post*sq_diff), axis=0)           # (K)
    denominator = post.T @ d_Cu                          # (K)
    var_hat = np.maximum(numerator / denominator, min_variance)  # (K)

    return GaussianMixture(mu_hat, var_hat, p_hat)

def mstep_compact(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    
    n, d = X.shape
    mu, _, _ = mixture
    K, _ = mu.shape
    p_hat = np.sum(post, axis=0) / n           # (K)

    # Cu is a mask of the observed values per point x cluster
    mask = (X != 0)                                          # (n, d)
    Cu = np.broadcast_to(mask[:, np.newaxis, :], (n, K, d))  # (n, K, d)
    d_Cu = np.sum(mask, axis=1)                              # (n)

    numerator = post.T @ X
    denominator = (post.T @ mask)  # (K, d)
    mu_raw = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator >= 1))
    mu_hat = np.where(denominator >= 1, mu_raw, mu)  # (K, d)

    var_hat = np.maximum(np.sum((post*np.sum(((X[:, np.newaxis, :] - mu_hat[np.newaxis, :, :])**2) * Cu, axis=2)), axis=0)\
                          / (post.T @ d_Cu), min_variance)   # (K)

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
      post, log_likelihood = estep_compact(X, mixture)
      mixture = mstep_compact(X, post, mixture)
    return (mixture, post, log_likelihood)

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    X_pred = X.copy()
    K, _ = mixture.mu.shape

    # e-step again -----------------------------------------
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
    # ------------------------------------------------------

    for i in range(n):
        for l in range(d):
            if X[i, l] == 0:
                X_pred[i, l] = np.dot(post[i], mixture.mu[:, l])

    return X_pred

def rmse(X_pred, X_gold):
    return np.sqrt( np.sum((X_pred - X_gold)**2) / X_gold.size)