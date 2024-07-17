import numpy as np
import pandas as pd

data = pd.read_csv('./mnist_train.csv/mnist_train.csv')

data.head()
data = np.array(data)

row, col = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:col]

data_train = data[1000:row].T
Y_train = data_train[0]
Y_train = data_train[1:col]

def init_params():
    W1 = np.random.randn(10, 784) - 0.5
    b1 = np.random.randn(10, 1) - 0.5
    W2 = np.random.randn(10, 10) - 0.5
    b2 = np.random.randn(10, 1) - 0.5
    return W1, b1, W2, b2
def ReLU(Z):
    return np.maximum(0, Z)
def softMax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))
def forward(W1, b1, W2, b2, X):
    pass
    