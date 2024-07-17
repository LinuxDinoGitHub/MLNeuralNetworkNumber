import math
import numpy as np

layer_output = [[4.8, 0.25, -1.56],
                [1.4, -1.61, -2.03],
                [1.051, 2.031, -4.8]]

exp_values = np.exp(layer_output)


norm_values = exp_values / np.sum(layer_output, axis=1, keepdims=True)
print(norm_values,np.sum(norm_values))