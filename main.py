import numpy as np
inputs = [1, 2, 3, 2.5]
weights = [[0.4, 0.5, -0.2, 0.1],
          [0.2, -0.4, 0.6, 0.9],
           [-0.1, -0.6, 0.4, 0.1]]
bias = 2

output = np.dot(weights, inputs)
print(output)