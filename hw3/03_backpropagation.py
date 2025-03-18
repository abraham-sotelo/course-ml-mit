#!python

# Simple network
import numpy as np

x  =  3  # input
t  =  1  # target
# Parameters
w1 =  0.01
w2 = -5
b  = -1

# Layers
z1 = w1 * x
a1 = max(0, z1)
z2 = w2*a1 + b
y = 1/(1 + np.exp(-z2))

# Error
C = ((y - t)**2) / 2

# Backpropagation
d2 = (y-t)*np.exp(-z2)/(1+np.exp(-z2))**2
  # dC/dw2  
dc_dw2 = a1*d2
  # da1/dz1
da1 = 1 if z1 > 0 else 0
d1 = w2*d2*da1
  # dC/dw1
dc_dw1 = x*d1


print(f"Error = {C}")
print(f"delta2 = {d2}")
print(f"dc_dw2 = {dc_dw2}")
print(f"dc_dw1 = {dc_dw1}")
