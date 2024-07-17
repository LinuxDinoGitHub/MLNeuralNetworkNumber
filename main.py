import numpy as np
import pandas as pd

data = pd.read_csv('./mnist_train.csv/mnist_train.csv')

data.head()
data = np.array(data)

m, n = data.shape 
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
X_train = data_train[0]
Y_train = data_train[1:n]