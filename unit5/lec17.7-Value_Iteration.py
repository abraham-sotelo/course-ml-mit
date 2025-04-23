#!python
import numpy as np
from datetime import datetime

gamma = 0.5
R = np.array([0., 0., 0., 0., 1.])
V = np.zeros(5, dtype=float)

# Transition matrix T(s, a, s')
T = np.array([
  [[ .5,  .5,  .0,  .0,  .0], [2/3, 1/3,  .0,  .0,  .0], [ .5,  .5,  .0,  .0,  .0]],  # S_1
  [[.25,  .5, .25,  .0,  .0], [ .0, 2/3, 1/3,  .0,  .0], [1/3, 2/3,  .0,  .0,  .0]],  # S_2
  [[ .0, .25,  .5, .25,  .0], [ .0,  .0, 2/3, 1/3,  .0], [ .0, 1/3, 2/3,  .0,  .0]],  # S_3
  [[ .0,  .0, .25,  .5, .25], [ .0,  .0,  .0, 2/3, 1/3], [ .0,  .0, 1/3, 2/3,  .0]],  # S_4
  [[ .0,  .0,  .0,  .5,  .5], [ .0,  .0,  .0,  .5,  .5], [ .0,  .0,  .0, 1/3, 2/3]]   # S_5
# Actions   stay                     move right                 move left
], dtype=float)

print("Naive ---------------------------")
start_time = datetime.now()
for k in range(100):
  V_k = V.copy()
  for s in range(len(V)):
    P = np.zeros(3, dtype=float)
    for a in range(3):
      P[a] = R[s] + gamma * np.sum(T[s, a] * V_k)
    V[s] = max(P)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print(V)

print("Dot product ---------------------")
V = np.zeros(5, dtype=float)
start_time = datetime.now()
for k in range(100):
  V_k = V.copy()
  for s in range(len(V)):
    P = np.zeros(3, dtype=float)
    for a in range(3):
      P[a] = R[s] + gamma * np.dot(T[s, a], V_k)
    V[s] = max(P)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print(V)

print("Precompute R[s] ------------------")
V = np.zeros(5, dtype=float)
start_time = datetime.now()
for k in range(100):
  V_k = V.copy()
  for s in range(len(V)):
    expected_values = gamma*(T[s] @ V_k)
    V[s] = np.max(R[s] + expected_values)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print(V)

print("Matrix multiplications -----------")
V = np.zeros(5, dtype=float)
start_time = datetime.now()
r_res = R[:, None]
for k in range(100):
  V_k = V.copy()
  Q = r_res + gamma * np.einsum('sat,t->sa', T, V_k)
  V = np.max(Q, axis=1)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print(V)
