#!python

#Linear separability after first layer
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def layer():
  x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
  
  w01 = np.array([0, 1, 1])
  w02 = np.array([0, 1, 1])
  
  wx1 = np.array([0, 2, -2])
  wx2 = np.array([0, -2, 2])
  
  res = []
  
  for i in range(3):
    f = np.zeros(x.shape)
    for j in range(x.shape[0]):
      z1 = w01[i] + (wx1[i]*x[j][0] + wx1[i]*x[j][1])
      z2 = w02[i] + (wx2[i]*x[j][0] + wx2[i]*x[j][1])
  
      f[j][0] = 2*z1-3
      f[j][1] = 2*z2-3
    
    res.append(f)
  return res



def layer2():
  x = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
  
  w01 = 1
  w02 = 1
  
  w11 = 1
  w21 = -1
  w12 = -1
  w22 = 1
  
  activation = [lambda x:5*x-2, lambda x: np.max([0,x]), np.tanh, lambda x:x]
  res = []
  for i in range(len(activation)):
    f = np.zeros(x.shape)
    for j in range(x.shape[0]):
      z1 = w01 + (w11*x[j][0] + w21*x[j][1])
      z2 = w02 + (w12*x[j][0] + w22*x[j][1])
  
      f[j][0] = activation[i](z1)
      f[j][1] = activation[i](z2)
    
    res.append(f)
  return res



def plotf(f):
  plt.scatter(f[0][0], f[0][1], color="blue", marker='o', s = 100, label="x1")
  plt.scatter(f[1][0], f[1][1], color="red", marker='x', s = 100, label="x2")
  plt.scatter(f[2][0], f[2][1], color="red", marker='x', s = 100, label="x3")
  plt.scatter(f[3][0], f[3][1], color="blue", marker='o', s = 100, label="x4")
  plt.show()

print(f'resultado: {layer()}' )