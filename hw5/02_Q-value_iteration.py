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

''' Reward function R(s, a, s'):
If s == 0:
  - Rewards are 0 (no rewards are defined for state 0).
If s ≠ 0:
  - If s == s' (stay in the same state):
    - Reward is (s + 4)^(-1/2)
  - If s ≠ s' (transition to a different state):
    - Reward is |s' - s|^(1/3) '''
# Adding an extra dimension to compare s and s' combinations
S  = np.arange(N_STATES).reshape(-1, 1, 1)  # State
Sn = np.arange(N_STATES).reshape(1, 1, -1)  # Next state
# Create two masks to control the values or R, embedding s ≠ 0 on them
mask_same = (S != 0) & (S == Sn)                   # (6,1,6)
mask_diff = (S != 0) & (S != Sn)
# Easier to get the reward by multiplying with the masks than masking on R
reward_same = mask_same * (S + 4)**(-1/2)          # (6,1,6)
reward_diff = mask_diff * np.abs(Sn - S)**(1/3)
# Masks are mutually exclusive, so I can add them
rewards = reward_same + reward_diff                # (6,1,6)
# Numpy can broadcast to (6,2,1) by adding the rewards to 0(6,2,6)
R = np.zeros(T.shape) + rewards                    # (6,2,6)
# Make reward = zero for the invalid transitions
R[T == 0] = 0

Q = np.zeros([N_STATES, N_ACTIONS])
Q_k = Q.copy()
Q = Q_k + np.sum(T * (R + gamma*np.max(Q, axis=1)), axis = 2)
V = np.max(Q, axis=1)
Policy = np.argmax(Q, axis=1)

print(Q)
print(V)
print(Policy)