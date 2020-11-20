from keras import backend as k
import numpy as np

a = [[1,2,3,4,5,6],[1,2,3,4,5,6]]
inputs = np.array(a)


k.mean(inputs, axis = -1,keepdims = True)


print(inputs)