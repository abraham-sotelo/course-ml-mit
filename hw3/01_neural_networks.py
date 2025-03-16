import numpy as np

def neural():
  # Activation functions
  f = np.vectorize(lambda a: np.max([a, 0]))
  s = np.vectorize(lambda a,b: np.exp(a) / (np.exp(a) + np.exp(b)))

  # Input
  x = np.array([[3, 14]])

  #Parameters
  W = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
  Wo = np.array([-1, -1, -1, -1])
  V = np.array([[1, -1], [1, -1], [1, -1], [1, -1]])
  Vo = np.array([[0, 2]])

  # Hidden layer
  z = (x @ W) + Wo
  fz = f(z)

  # Output layer
  u = (fz @ V) + Vo
  fu = f(u).T
  o1 = s(fu[0], fu[1])
  o2 = s(fu[1], fu[0])
  print(f"Before softmax: fu1={fu[0]}   fu2={fu[1]}")

  return o1, o2

o = neural()
print(f"o1 = {o[0]}")
print(f"o2 = {o[1]}")