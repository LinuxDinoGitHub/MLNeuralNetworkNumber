import numpy as np

np.random.seed(0)
X = [[1, 2, 3, 2.5],
    [2, 6, 5, -3],
    [3.3, 2.6, -1.9, 2.2]]

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
layer1 = Layer(4,3)
layer2 = Layer(3,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)