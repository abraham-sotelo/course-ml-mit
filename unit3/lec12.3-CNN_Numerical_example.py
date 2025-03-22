import numpy as np
import scipy as sci
from skimage.measure import block_reduce

I = np.array([[1,0,2],[3,1,0],[0,0,4]])  # Input
F = np.array([[1,0],[0,1]])              # Filter

conv = sci.signal.convolve2d(I, F, mode="valid")
relu = np.maximum(conv,0)
pool = block_reduce(relu, block_size=(2, 2), func=np.max)
print(f"The output of the CNN is: {pool.item()}")