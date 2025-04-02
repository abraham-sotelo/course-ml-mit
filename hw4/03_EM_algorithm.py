'''
Data log-likehood

p(x,theta) = pi_1*N + pi_2*N

'''
import numpy as np

# Functions definitions
gaussian = lambda x, mu, va, d: np.exp(((x - mu)**2)/(-2 * va)) / ((2 * np.pi * va)**(d/2))
mixture = lambda pi, N: np.sum(pi * N, axis=1)
log_likelihood = lambda p: np.sum(np.log(p))
joint_P = lambda pi, N: pi * N

# Input
x = np.array([-1,0,4,5,6])
n = len(x)
d = 1
x = x.reshape(n, d)

# Initial parameters
pi = np.array([0.5, 0.5])
mean = np.array([6, 7])
var = np.array([1, 4])

# get the data log-likelihood
N = gaussian(x, mean,var,d)
p = mixture(pi, N)
l = log_likelihood(p)
print(f"data log-likekihood: {l:.1f}")

# E-step
J = joint_P(pi, N)
P = J / p.reshape(n,d)

# Cluster classification
C = np.where(P[:, 0] > P[:, 1], 1, 2)
print(f"Cluster classification: {C}")

# M-step
expected_n = np.sum(P, axis=0)
prior = expected_n / n
mean = np.sum(P * x, axis=0) / expected_n
variance = np.sum(P * (x - mean)**2, axis=0) / (expected_n * d)

print("Updated θ")
for j in range(len(mean)):
  print(f"prior_{j+1} = {prior[j]:.5f}")
  print(f"mean_{j+1} = {mean[j]:.5f}")
  print(f"variance_{j+1} = {variance[j]:.5f}")