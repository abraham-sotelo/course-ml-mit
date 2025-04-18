"""Mixture model for matrix completion using matrix operations"""
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

    diff = (X[:,np.newaxis,:] - mu[np.newaxis,:,:])
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