import numpy as np
import random as r
import tensorflow as tf

a = np.array([[[[5], [6], [7]], [[5], [6], [7]], [[5], [6], [7]]]])
b = np.array([[[[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]]]])
d = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]]])

print('a.shape :\n', a.shape)
print('b.shape :\n', b.shape)
print('d.shape :\n', d.shape)
c = np.concatenate([a, b], axis=3)
print('c : \n', c)
print('c.shape :\n', c.shape)