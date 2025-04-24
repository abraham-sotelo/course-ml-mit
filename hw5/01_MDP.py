#!python

import numpy as np

V = np.zeros(4, dtype=float)
R = np.array([
  [ 1.,  0.], # A
  [ 1.,  1.], # B
  [10.,  1.], # C
  [ 0., 10.]  # D
#   UP DOWN  
])
gamma = np.float64(0.75)

A = 0
B = 1
C = 2
D = 3

UP = 0
DOWN = 1

n_states = len(V)

for k in range(1000):
  V_k = V.copy()
  for s in range(n_states):
    if s == A:
      V[s] = R[A, UP] + gamma*V_k[B]
    elif s == B or s == C:
      Q_UP = R[s, UP] + gamma*V_k[s+1]
      Q_DOWN = R[s, DOWN] + gamma*V_k[s-1]
      V[s] = np.maximum(Q_UP, Q_DOWN)
    else:
      V[s] = R[D, DOWN] + gamma*V_k[C]
    
  if np.array_equal(V, V_k):
    print(f"Iterations to convergence: {k}")
    break

print(V)