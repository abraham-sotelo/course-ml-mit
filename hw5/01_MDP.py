#!python

import numpy as np
from datetime import datetime

n_states = 4
Vo = np.zeros(n_states, dtype=float)
R = np.array([
  [ 1.,  0.], # A
  [ 1.,  1.], # B
  [10.,  1.], # C
  [ 0., 10.]  # D
#   UP DOWN
])

gamma = np.float64(0.75)
K = 1000

A = 0
B = 1
C = 2
D = 3

UP = 0
DOWN = 1

def naive():
  V = Vo.copy()

  for k in range(K):
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
      convergence = k
      break
  return (V, convergence)


def vectorized():
  V = Vo.copy()
  # Transition table: T[s, a] = s'
  T = np.array([
    [B, A],  # A: UP->B, DOWN invalid -> A
    [C, A],  # B: UP->C, DOWN->A
    [D, B],  # C: UP->D, DOWN->B
    [D, C],  # D: UP invalid -> D, DOWN->C
  ], dtype=int)

  for k in range(K):
    V_k = V.copy()
    Q = R + gamma * V_k[T]  # V[T] = V(s') Value next stage
    V = np.max(Q, axis=1)
    if np.array_equal(V, V_k):
      convergence = k
      break
  return (V, convergence)


start_time = datetime.now()
V, k = naive()
end_time = datetime.now()
print(f"V* = {V}")
print(f'Iterations to convergence: {k}')
print('Duration naive: {}'.format(end_time - start_time))
print("---------------------")
start_time = datetime.now()
V, k = vectorized()
end_time = datetime.now()
print(f"V* = {V}")
print(f'Iterations to convergence: {k}')
print('Duration vectorized: {}'.format(end_time - start_time))
