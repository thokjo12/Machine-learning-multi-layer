import numpy as np

import fcnn
import mnist
import os

import utils

def weight_initialization(output_units,input_units):
	weight_shape = (output_units, input_units)
	return np.random.uniform(-1, 1, weight_shape)


if not os.path.exists("data/mnist.pkl"):
    mnist.init()

n_classes = 10
units_second_layer = 64

X_train, Y_train, X_test, Y_test = mnist.load()
X_train, Y_train, X_val, Y_val = utils.create_validation_data(X_train, Y_train, percentage_train=0.9)

X_train = (X_train / 127.5) - 1.0
X_test = (X_test / 127.5) - 1.0
X_val = (X_val / 127.5) - 1.0

X_train = utils.bias_trick(X_train)
X_test = utils.bias_trick(X_test)
X_val = utils.bias_trick(X_val)

# normalize data to [-1,1]

Y_test = utils.onehot_encode(Y_test, n_classes)
Y_train = utils.onehot_encode(Y_train, n_classes)

# weights from input to hidden layer ((28x28+1,Unites_second_layer)
w_ji = weight_initialization(units_second_layer,X_train.shape[1])
# weights from hidden to output (Unites_second_layer,Classes)
w_kj = weight_initialization(n_classes,units_second_layer)
print(X_test.min(),X_test.max())
print(X_train.min(),X_train.max())
print(X_val.min(),X_val.max())

a_k, a_j = fcnn.forward_pass(w_ji=w_ji,w_kj=w_kj,x=X_train)
w_ji,w_kj=fcnn.sgd(a_k=a_k[0:4],a_j=a_j[0:4],a_i=X_train[0:4],targets=Y_train[0:4],w_kj=w_kj,w_ji=w_ji,lr=0.01,check_grad=False)
print(w_ji.shape, w_kj.shape)

