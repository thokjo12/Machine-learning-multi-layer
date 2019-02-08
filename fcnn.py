import numpy as np


def accuracy(target, output):
    """
    Takes the argmax of targets, which returns only the indexes that are one for axis 1,
    Assuming that the highest value in axis=1 for output is its prediction we can take
    argmax of output as well.
    np.equal works element wise so we get a matrix of all the correct predictions related to target
    then we can take the mean of all these predictions relative to the total set which is our decimal accuracy
    :param target: the ground truth labels
    :param output: the predicted labels
    :return: the accuracy of the prediction for all labels
    """
    return np.mean(np.equal(target.argmax(axis=1), output.argmax(axis=1)))


def cross_entropy_loss(output, targets, eps=1e-15):
    cost = np.sum(targets * np.log(output + eps), axis=1)
    return - np.mean(cost)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(a):
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)


def forward_pass(w_kj, w_ji, x):
    """
    forward pass through all the layers in the network
    :param w_kj: weights from hidden to output
    :param w_ji: weights from input to hidden
    :param x: the input values
    :return: outputs the activation of each layer
    """
    z_j = x.dot(w_ji.T)
    a_j = sigmoid(z_j)

    z_k = a_j.dot(w_kj.T)
    a_k = softmax(z_k)
    return a_k, a_j


def sgd(a_k, a_j, a_i, targets, w_kj, w_ji, lr, check_grad):
    """
    :param a_k: output from the output layer
    :param a_j: output from the hidden layer
    :param a_i: the network input
    :param targets: the truth labels
    :param w_kj: weights from hidden to output
    :param w_ji:  weights from input to hidden
    :param lr: learning rate
    :param check_grad: boolean val for checking gradients
    :return: new weigth corrections
    """

    d_k = -(targets - a_k)
    d_j = (a_j * (1 - a_j)) * d_k.dot(w_kj)

    grad_kj = (d_k[:, np.newaxis, :] * a_j[:, :, np.newaxis]).mean(axis=0).T
    grad_ji = (d_j[:, np.newaxis, :] * a_i[:, :, np.newaxis]).mean(axis=0).T

    if check_grad:
        check_gradient(a_i, targets, w_ji, w_kj, 1e-2, grad_ji, grad_kj)

    w_kj = w_kj - lr * grad_kj
    w_ji = w_ji - lr * grad_ji

    return w_ji, w_kj


def check_gradient(a_i, targets, w_ji, w_kj, epsilon, grad_ji, grad_kj):
    print("Checking gradient...")

    dw_kj = np.zeros_like(w_kj)
    for k in range(w_kj.shape[0]):
        for j in range(w_kj.shape[1]):
            new_kj1, new_kj2 = np.copy(w_kj), np.copy(w_kj)
            new_kj1[k, j] += epsilon
            new_kj2[k, j] -= epsilon

            out1, _ = forward_pass(new_kj1, w_ji, a_i)
            out2, _ = forward_pass(new_kj2, w_ji, a_i)

            loss1 = cross_entropy_loss(out1, targets)
            loss2 = cross_entropy_loss(out2, targets)

            dw_kj[k, j] = (loss1 - loss2) / (2 * epsilon)

    maximum_abosulte_difference1 = abs(grad_kj - dw_kj).max()
    assert maximum_abosulte_difference1 <= epsilon ** 2, "Absolute error was: {}".format(maximum_abosulte_difference1)

    print("Gradient for kj is valid.")
    print("Absolute error for kj was: {}".format(maximum_abosulte_difference1))

    dw_ji = np.zeros_like(w_ji)
    for j in range(w_ji.shape[0]):
        for i in range(w_ji.shape[1]):
            new_ji_1, new_ji_2 = np.copy(w_ji), np.copy(w_ji)
            new_ji_1[j, i] += epsilon
            new_ji_2[j, i] -= epsilon

            out3, _ = forward_pass(w_kj, new_ji_1, a_i)
            out4, _ = forward_pass(w_kj, new_ji_2, a_i)

            loss3 = cross_entropy_loss(out3, targets)
            loss4 = cross_entropy_loss(out4, targets)

            dw_ji[j, i] = (loss3 - loss4) / (2 * epsilon)

    maximum_abosulte_difference2 = abs(grad_ji - dw_ji).max()
    assert maximum_abosulte_difference2 <= epsilon ** 2, "Absolute error was: {}".format(maximum_abosulte_difference2)

    print("Gradient for ji is valid.")
    print("Absolute error for ji was: {}".format(maximum_abosulte_difference2))


def fit(w_kj, w_ji, epochs, batches_per_epoch):
    for epoch in range(epochs):
        for iteration in range(batches_per_epoch):
            pass

            # lr = 0.01
            # def softmax(a):
            #     a_exp = np.exp(a)
            #     return a_exp / a_exp.sum(axis=1, keepdims=True)
            #
            #
            # w_ji = np.zeros((784, 64))
            # w_kj = np.zeros((64, 10))
            #
            # x_i = np.zeros((32, 784))
            # truth = np.random.random((1, 10))
            # a_j = -x_i.dot(w_ji)
            # a_k = a_j.dot(w_kj)
            # dk = (-(truth - a_k))
            # dj = np.multiply((a_j * (1 - a_j)),dk.dot(w_kj.T))
            # w_kj_new = w_kj - lr * (a_j.T.dot(dk))
            # w_ji_new = w_ji - lr *x_i.T.dot(dj)
            # print("input shapes: xi{} w_ji{} w_kj{}".format(x_i.shape,w_ji.shape,w_kj.shape))
            # print("sig shape in hidden layer: ",a_j.shape)
            # print("softmax shape in output layer: ",a_k.shape)
            # print("shape of delta rule in output:",dk.shape)
            # print("shape of delta rule in hidden",dj.shape)
            # print("new w_ji {} new w_kj {}".format(w_ji.shape,w_kj.shape))
