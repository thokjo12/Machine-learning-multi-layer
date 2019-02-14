import numpy as np
import matplotlib.pyplot as plt
import task4
import mnist
import os

import utils

if not os.path.exists("data/mnist.pkl"):
    mnist.init()

n_classes = 10
units_second_layer = 59
units_third_layer = 59

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

bound_w_ji = 1 / np.sqrt(X_train.shape[1])
bound_w_kj = 1 / np.sqrt(units_second_layer)
bound_w_mk = 1 / np.sqrt(units_third_layer)

# weights from input to hidden layer ((28x28+1,Unites_second_layer)
w_ji = np.random.normal(0, bound_w_ji, (units_second_layer, X_train.shape[1]))

# weights from hidden to output (Unites_second_layer,Classes)
w_kj = np.random.normal(0, bound_w_kj, (units_second_layer, units_second_layer))
w_mk = np.random.normal(0, bound_w_mk, (n_classes, units_second_layer))

w_ji, w_kj, w_mk, meta = task4.fit(x_train=X_train, y_train=Y_train,
                                   x_val=X_val, y_val=Y_val,
                                   x_test=X_test, y_test=Y_test,
                                   w_kj=w_kj, w_ji=w_ji, w_mk=w_mk,
                                   epochs=15, check_step_divisor=10,
                                   batch_size=32, initial_lr=0.05,
                                   lr_decay=None, my=0.2, check_grad=False)

test_final_a_m, test_final_a_k, test_final_a_j, = task4.forward_pass(w_kj, w_ji, w_mk, X_test)
final_test_loss = task4.cross_entropy_loss(test_final_a_m, Y_test)
final_test_accuracy = task4.accuracy(Y_test, test_final_a_m)

validation_final_a_m, validation_final_a_k, validation_final_a_j, = task4.forward_pass(w_kj, w_ji, w_mk, X_val)
final_validation_loss = task4.cross_entropy_loss(validation_final_a_m, Y_val)
final_validation_accuracy = task4.accuracy(Y_val, validation_final_a_m)

train_final_a_m, train_final_a_k, train_final_a_j, = task4.forward_pass(w_kj, w_ji, w_mk, X_train)
final_train_loss = task4.cross_entropy_loss(train_final_a_m, Y_train)
final_train_accuracy = task4.accuracy(Y_train, train_final_a_m)
print("final training loss for training during training: {}".format(meta["train_loss"][-1]))
print("final validation loss for validation during training: {}".format(meta["val_loss"][-1]))
print("Test loss on the test set: {}".format(final_test_loss))
print("Final accuracy on the test set:", final_test_accuracy)
print("Final accuracy on the train set:", final_train_accuracy)
print("Final accuracy on the validation set:", final_validation_accuracy)

plt.figure(figsize=(12, 8))
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(meta["step"], meta["train_loss"], label="Training loss")
plt.plot(meta["step"], meta["val_loss"], label="valdiation loss")
plt.plot(meta["step"], meta["test_loss"], label="test loss")
plt.plot(meta["step"][-1], final_test_loss, ".", label="Final test loss")
plt.legend()

# plot accuracy
plt.figure(figsize=(12, 8))
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.plot(meta["step"], meta["train_acc"], label="Training acc")
plt.plot(meta["step"], meta["val_acc"], label="valdiation acc")
plt.plot(meta["step"], meta["test_acc"], label="test acc")
plt.legend()
plt.show()
