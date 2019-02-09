import time

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
    """
    gives us a measure on how accurate the model predictions are
    sums all targets and the log of the output into a NX1 matrix which gives us
    the loss for each of these samples, then we can take the average and we get the average loss for the whole set.
    :param eps: a small variable to prevent 0 in log
    :param targets: the labels
    :param output: the predicted labels
    :return: the average loss over a batch
    """
    # sum
    # cost = np.sum(targets * np.log(output))
    cost = targets * np.log(output+eps)
    return - np.mean(cost)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """
    takes a vector of dimension (N,NUM_CLASSES) and forms a normalized probability vector with the length of NUM_CLASSES
    each probability vector/classification vector should sum to 1, in some cases it will not because of floating point
    rounding errors.
    :param z: the forward pass output
    :return: a NXNUM_CLASSES matrix depicting the probabilities for N samples relative to NUM_CLASSES.
    """
    a_exp = np.exp(z)
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


def sgd(a_k, a_j, a_i, targets, w_kj, w_ji, lr, check_grad,norm_factor):
    """
    :param a_k: output from the output layer
    :param a_j: output from the hidden layer
    :param a_i: the network input
    :param targets: the truth labels
    :param w_kj: weights from hidden to output
    :param w_ji:  weights from input to hidden
    :param lr: learning rate
    :param check_grad: boolean val for checking gradients
    :return: new weight corrections
    """
    d_k = -(targets - a_k)
    d_j = (a_j * (1 - a_j)) * d_k.dot(w_kj)

    grad_kj = d_k.T.dot(a_j) / norm_factor
    grad_ji = d_j.T.dot(a_i) / norm_factor
    # double check these as they cost too much to perform but seem to have more "realistic values"
    # grad_kj = (d_k[:, np.newaxis, :] * a_j[:, :, np.newaxis]).mean(axis=0)  # dk -> (n,1,10) a_j -> (n,64,1)
    # grad_ji = (d_j[:, np.newaxis, :] * a_i[:, :, np.newaxis]).mean(axis=0)  # dj -> (n,1,64) a_i -> (n,785,1)
    if check_grad:
        check_gradient(a_i, targets, w_ji, w_kj, 1e-2, grad_ji, grad_kj)

    w_kj = w_kj - lr * grad_kj
    w_ji = w_ji - lr * grad_ji

    return w_ji, w_kj


def check_gradient(a_i, targets, w_ji, w_kj, epsilon, grad_ji, grad_kj):
    """
    checking of gradients with numerical approximation of the function gradient.
    :param a_i: network input
    :param targets: the target labels
    :param w_ji: weights from input to hidden layer
    :param w_kj: weights from hidden to output
    :param epsilon: our error "space"
    :param grad_ji: the gradient function for input -> hidden
    :param grad_kj: the gradient function for hidden -> output
    :return: no return
    """
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


def fit(x_train, y_train, x_val, y_val, x_test, y_test, w_kj, w_ji, epochs, check_step, batch_size, lr,
        check_grad=False):
    batches_per_epoch = batch_size * 10
    check_step = batches_per_epoch // check_step
    norm_factor = (batch_size * 10)
    STEP = []

    TRAIN_LOSS = []
    VAL_LOSS = []
    TEST_LOSS = []

    TEST_ACC = []
    VAL_ACC = []
    TRAIN_ACC = []

    iteration = 0
    for epoch in range(epochs):
        print(epoch)
        for i in range(batches_per_epoch):
            iteration += 1

            x_batch = x_train[i * batch_size:(i + 1) * batch_size]
            y_batch = y_train[i * batch_size:(i + 1) * batch_size]

            a_k, a_j = forward_pass(w_kj, w_ji, x_batch)
            w_ji, w_kj = sgd(a_k, a_j, x_batch, y_batch, w_kj, w_ji, lr, check_grad,norm_factor)

            if i % check_step == 0:
                #
                STEP.append(iteration)
                a_k_train, a_j_train = forward_pass(w_kj, w_ji, x_train)
                a_k_val, a_j_val = forward_pass(w_kj, w_ji, x_val)
                a_k_test, a_j_test = forward_pass(w_kj, w_ji, x_test)

                TEST_ACC.append(accuracy(y_test, a_k_test))
                VAL_ACC.append(accuracy(y_val, a_k_val))
                TRAIN_ACC.append(accuracy(y_train, a_k_train))

                TRAIN_LOSS.append(cross_entropy_loss(a_k_train, y_train))
                VAL_LOSS.append(cross_entropy_loss(a_k_val, y_val))
                TEST_LOSS.append(cross_entropy_loss(a_k_test, y_test))

    return w_ji, w_kj, {
        "val_loss": VAL_LOSS,
        "train_loss": TRAIN_LOSS,
        "test_loss": TEST_LOSS,
        "test_acc": TEST_ACC,
        "val_acc": VAL_ACC,
        "train_acc": TRAIN_ACC,
        "step": STEP}
