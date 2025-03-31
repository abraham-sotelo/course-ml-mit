#!python
'''
Asume that the initial means and variances of two clusters in a GMM are as follows
μ1 = -3       σ^2_1 = σ^2_2 = 4       p1 = p2 = 0.5
μ2 = -2

Let X = {0.2, -0.9, -1, 1.2, 1.8}

In this problem we compute the updated parameters corresponding to cluster 1
Compute the posterior probabilities
Compute the updated parameters corresponding
'''

import numpy as np

# Function definitions ----------------------------------
gaussian = lambda X, Mu, Sig, d: np.exp(((X - Mu)**2) / (-2 * Sig)) / ((2 * np.pi * Sig)**(d/2))  #[n,k]
joint_P = lambda G, P: G * P                  #[n,k]
likelihood = lambda J: np.sum(J, axis = 1).reshape(len(J),1)  #[5,1]

# Initialization ------------------------------------------
# Observer data
data = np.array([[0.2, -0.9, -1, 1.2, 1.8]])
d, n = data.shape
data = data.reshape(n,d)

# Initial parameters
mean     = np.array([[-3, 2]])
variance = np.array([[4, 4]])
prior    = np.array([[0.5, 0.5]])

# Sanity check 
assert mean.shape[0] == data.shape[1]
assert variance.shape == mean.shape
assert prior.shape == variance.shape

# E-step --------------------------------------------------
G = gaussian(data, mean, variance, d)
J = joint_P(G, prior)
posterior = J / likelihood(J)

for j, i in np.ndindex(posterior.T.shape): # transpose to printer nicer
  print(f"P({j+1}|{i+1}) = {posterior[i][j]:.5f}")

# M-step --------------------------------------------------
expected_n = np.sum(posterior, axis=0)
prior = expected_n / n
mean = np.sum(posterior * data, axis=0) / expected_n
variance = np.sum(posterior * (data - mean)**2, axis=0) / (expected_n * d)

print("Updated θ")
for j in range(len(mean)):
  print(f"prior_{j+1} = {prior[j]:.5f}")
  print(f"mean_{j+1} = {mean[j]:.5f}")
  print(f"variance_{j+1} = {variance[j]:.5f}")