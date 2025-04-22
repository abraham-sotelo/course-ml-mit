#!python
import numpy as np

gamma = 0.5
R = np.array([0., 0., 0., 0., 1.])
V = np.zeros(5)

for k in range(3):
  for s in range(len(V)):
    if s == 4:
      if k == 0:
        V[s] = R[s] + gamma*V[s]
      continue
    V[s] = R[s] + gamma*V[s+1]

print(V)