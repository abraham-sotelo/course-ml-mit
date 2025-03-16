#!python
import numpy as np

def lstm(x):
  # Input x, 0 <= t <= n

# Parameters
  Wfh = 0
  Wih = 0
  Woh = 0

  Wfx = 0
  Wix = 100
  Wox = 100

  bf = -100
  bi = 100
  bo = 0

  Wch = -100
  Wcx = 50
  bc = 0
  # ----------

  sigmoid = lambda a: 1/(1 + np.exp(-a))  # Activation function

  # For simplicity, the indices of x and (c, h) are not the same
  # h[0] is the initial state corresponding with x[-1]
  # h[t] is the previous state, corresponding with x[t-1]
  # h[t+1] is the current state, corresponding with x[t]
  n = len(x)
  h = np.zeros(n + 1)  # Visible state -1 <= t <= n
  c = np.zeros(n + 1)  # Memory cell   -1 <= t <= n 

  # LSTM Unit
  for t in range(n):
    # Gates
    ft = sigmoid(Wfh*h[t] + Wfx*x[t] + bf)
    it = sigmoid(Wih*h[t] + Wix*x[t] + bi)
    ot = sigmoid(Woh*h[t] + Wox*x[t] + bo)

    # States
    c[t+1] = ft*c[t] + it*np.tanh(Wch*h[t] + Wcx*x[t] + bc)
    h[t+1] = np.round(ot*np.tanh(c[t+1]))

  return(h[1:]) 


print(f"Visible states 1: {lstm(np.array([0,0,1,1,1,0]))}")
print(f"Visible states 2: {lstm(np.array([1,1,0,1,1]))}")