import numpy as np
import scipy as sci
# One dimensional

f = [1,3,-1,1,-3]
f_padded = np.pad(f, (1,1), 'constant', constant_values=0)
g = [-1,0,1]

h = np.convolve(f, g, mode="valid")
ho = np.convolve(f_padded, g, mode='valid')
print(h)
print(ho)


# Two dimensional
f = np.array([[1,2,1],[2,1,1],[1,1,1]])
g = np.array([[1, 0.5], [0.5, 1]])
flipped_g = np.flip(g, axis=(0,1))
h = sci.signal.convolve2d(f, flipped_g, mode="valid")

print(h)
print(h.sum())

