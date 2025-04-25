#!python

import numpy as np

N_STATES = 6
N_ACTIONS = 2
gamma = 0.6
T = np.array([                                         # States
  [[1., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]],  # 0
  [[1., 0., 0., 0., 0., 0.], [0., .3, 0., .7, 0., 0.]],  # 1
  [[0., 1., 0., 0., 0., 0.], [0., 0., .3, 0., .7, 0.]],  # 2
  [[0., 0., 1., 0., 0., 0.], [0., 0., 0., .3, 0., .7]],  # 3
  [[0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.]],  # 4
  [[0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]]   # 5
])
# Actions     M                          C

# Filling R
R = np.zeros(T.shape)
for s in range(N_STATES):
  for a in range(N_ACTIONS):
    for sp in range(N_STATES):
      if s != 0:
        if s == sp:
          R[s,a,sp] = (s + 4)**(-1/2)
        else:
          R[s,a,sp] = np.abs(sp - s)**(1/3)
R[T == 0] = 0

V = np.zeros(N_STATES)
Q = np.zeros([N_STATES, N_ACTIONS])

sample = R
Q_k = Q.copy()
Q = Q_k + np.sum(T * (R + gamma*np.max(Q, axis=1)), axis = 2)

V = np.max(Q, axis=1)
P = np.argmax(Q, axis=1) # Policy