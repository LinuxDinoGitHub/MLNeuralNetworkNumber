import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2, 6, 5, -3],
    [3.3, 2.6, -1.9, 2.2]]

def create_data_set(points, classes):
    X = np.zeros((points*classes, 2)) # data matrix (each row = single example)
    y = np.zeros(points*classes, dtype='uint8') # class labels
    for j in range(classes):
        ix = range(points*j,points*(j+1))
        r = np.linspace(0.0,1,points) # radius
        t = np.linspace(j*4,(j+1)*4,points) + np.random.randn(points)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

class Layer:
    def __init__(self, n_inputs, n_neurons): #4 inputs by 3 
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons) #3 by 4 matrix
        self.biases = np.zeros((1, n_neurons)) #3 list
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
layer1 = Layer(4,3)
layer2 = Layer(3,2)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
