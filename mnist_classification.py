import numpy as np
import mnist
import os

import utils

if not os.path.exists("data/mnist.pkl"):
    mnist.init()

n_classes = 10
units_second_layer = 64

X_train, Y_train, X_test, Y_test = mnist.load()
X_train, Y_train, X_val, Y_val = utils.create_validation_data(X_train, Y_train, percentage_train=0.9)

X_train = utils.bias_trick(X_train)
X_test = utils.bias_trick(X_test)
X_val = utils.bias_trick(X_val)

# normalize data to [-1,1]
X_train = (X_train / 127.5) - 1.0
X_test = (X_test / 127.5) - 1.0
X_val = (X_val / 127.5) - 1.0

Y_test = utils.one_hot_encode(Y_test, n_classes)
Y_train = utils.one_hot_encode(Y_train, n_classes)

# weights from input to hidden layer ((28x28+1,Unites_second_layer)
w_i_h = np.random.uniform([-0.5], [0.5], (X_train.shape[1], units_second_layer))
# weights from hidden to output (Unites_second_layer,Classes)
w_h_o = np.random.uniform([-0.5], [0.5], (units_second_layer, n_classes))



