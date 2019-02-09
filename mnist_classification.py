import numpy as np
import matplotlib.pyplot as plt
import multi_layer_network
import mnist
import os

import utils


def weight_initialization(output_units, input_units):
    weight_shape = (output_units, input_units)
    return np.random.uniform(-1, 1, weight_shape)


if not os.path.exists("data/mnist.pkl"):
    mnist.init()

n_classes = 10
units_second_layer = 64

X_train, Y_train, X_test, Y_test = mnist.load()
X_train, Y_train, X_val, Y_val = utils.create_validation_data(X_train, Y_train, percentage_train=0.9)

# normalize data to [-1,1]
X_train = (X_train / 127.5) - 1.0
X_test = (X_test / 127.5) - 1.0
X_val = (X_val / 127.5) - 1.0

X_train = utils.bias_trick(X_train)
X_test = utils.bias_trick(X_test)
X_val = utils.bias_trick(X_val)

Y_test = utils.onehot_encode(Y_test, n_classes)
Y_train = utils.onehot_encode(Y_train, n_classes)
Y_val = utils.onehot_encode(Y_val, n_classes)

# weights from input to hidden layer ((28x28+1,Unites_second_layer)
w_ji = weight_initialization(units_second_layer, X_train.shape[1])

# weights from hidden to output (Unites_second_layer,Classes)
w_kj = weight_initialization(n_classes, units_second_layer)


w_kj, w_ji, meta = multi_layer_network.fit(x_train=X_train, y_train=Y_train,
                                           x_val=X_val, y_val=Y_val,
                                           x_test=X_test, y_test=Y_test,
                                           w_kj=w_kj, w_ji=w_ji,
                                           epochs=15, check_step=20,
                                           batch_size=64, lr=0.5,
                                           check_grad=False)


plt.figure(figsize=(12, 8))
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(meta["step"], meta["train_loss"], label="Training loss")
plt.plot(meta["step"], meta["val_loss"], label="valdiation loss")
plt.plot(meta["step"], meta["test_loss"], label="test loss")
# plt.plot(meta["step"][-1], final_loss, ".", label="Final test loss")


#plot accuracy
plt.figure(figsize=(12, 8))
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(meta["step"], meta["train_acc"], label="Training acc")
plt.plot(meta["step"], meta["val_acc"], label="valdiation acc")
plt.plot(meta["step"], meta["test_acc"], label="test acc")
plt.legend()
plt.show()

# a_k, a_j = multi_layer_network.forward_pass(w_ji=w_ji, w_kj=w_kj, x=X_train)
# w_ji, w_kj = multi_layer_network.sgd(a_k=a_k[0:4], a_j=a_j[0:4], a_i=X_train[0:4], targets=Y_train[0:4], w_kj=w_kj, w_ji=w_ji,
#  lr=0.01,check_grad=True,norm_factor=4*10)
