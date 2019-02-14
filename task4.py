import numpy as np


def learning_rate_annealing(initial_lr, count, t_decay):
    """
    Causes a reduction in learning rate over time, if t_decay is large, almost no reduction will happen
    :param initial_lr: the learning rate of training/fit start
    :param count: the count/iteration/epoch
    :param t_decay: the strength to decay with
    :return: new learning rate
    """
    if t_decay is None:
        return initial_lr
    return initial_lr / (1 + count / t_decay)


def should_early_stop(validation_loss, num_steps=10):
    if len(validation_loss) < num_steps + 1:
        return False
    is_increasing = [validation_loss[i] <= validation_loss[i + 1] for i in range(-num_steps - 1, -1)]
    return sum(is_increasing) == len(is_increasing)


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
    cost = targets * np.log(output + eps)
    return - np.mean(cost)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def improved_sigmoid(z):
    return 1.7159 * np.tanh((2 / 3) * z)


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


def forward_pass(w_kj, w_ji, w_mk, x):
    """
    forward pass through all the layers in the network
    :param w_kj: weights from hidden to output
    :param w_ji: weights from input to hidden
    :param x: the input values
    :return: outputs the activation of each layer
    """
    z_j = x.dot(w_ji.T)
    a_j = improved_sigmoid(z_j)

    z_k = a_j.dot(w_kj.T)
    a_k = improved_sigmoid(z_k)

    z_m = a_k.dot(w_mk.T)
    a_m = softmax(z_m)
    return a_m, a_k, a_j


def sgd(a_k, a_j, a_i, a_m, targets, w_kj, w_ji, w_mk, lr, norm_factor, prev_update_kj, prev_update_ji,
        prev_update_mk, my=0):
    """
    :param w_mk:
    :param norm_factor:
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
    d_m = -(targets - a_m)
    d_k = ((1.7159 * 2.0) / (3.0 * (np.cosh((2 / 3) * a_j.dot(w_kj.T)) ** 2.0))) * d_m.dot(w_mk)
    d_j = ((1.7159 * 2.0) / (3.0 * (np.cosh((2 / 3) * a_i.dot(w_ji.T)) ** 2.0))) * d_k.dot(w_kj)

    grad_mk = d_m.T.dot(a_k) / norm_factor
    grad_kj = d_k.T.dot(a_j) / norm_factor
    grad_ji = d_j.T.dot(a_i) / norm_factor

    # if check_grad:
    #     check_gradient(a_i, targets, w_ji, w_kj, 1e-2, grad_ji, grad_kj)

    prev_update_kj = lr * grad_kj + my * prev_update_kj
    prev_update_mk = lr * grad_mk + my * prev_update_mk
    prev_update_ji = lr * grad_ji + my * prev_update_ji
    w_mk = w_mk - prev_update_mk
    w_kj = w_kj - prev_update_kj
    w_ji = w_ji - prev_update_ji

    return w_ji, w_kj, w_mk, prev_update_ji, prev_update_kj, prev_update_mk


STEP = []

TRAIN_LOSS = []
VAL_LOSS = []
TEST_LOSS = []

TEST_ACC = []
VAL_ACC = []
TRAIN_ACC = []


def fit(x_train, y_train, x_val, y_val, x_test, y_test, w_kj, w_ji, w_mk, epochs, check_step_divisor, batch_size,
        initial_lr,
        lr_decay, my, check_grad=False):
    meta = {"val_loss": VAL_LOSS, "train_loss": TRAIN_LOSS, "test_loss": TEST_LOSS, "test_acc": TEST_ACC,
            "val_acc": VAL_ACC, "train_acc": TRAIN_ACC, "step": STEP}

    batches_per_epoch = x_train.shape[0] // batch_size
    normalization_factor = (batch_size * y_train.shape[1])
    check_step = batches_per_epoch // check_step_divisor
    iteration = 0
    lr = initial_lr

    for epoch in range(epochs):
        items = np.arange(x_train.shape[0])
        np.random.shuffle(items)
        x_train = x_train[items]
        y_train = y_train[items]

        prev_update_mk = np.zeros_like(w_mk)
        prev_update_kj = np.zeros_like(w_kj)
        prev_update_ji = np.zeros_like(w_ji)
        for i in range(batches_per_epoch):
            iteration += 1

            x_batch = x_train[i * batch_size:(i + 1) * batch_size]
            y_batch = y_train[i * batch_size:(i + 1) * batch_size]

            a_m_out, a_k_out, a_j_out = forward_pass(w_kj, w_ji, w_mk, x_batch)
            w_ji, w_kj, w_mk, prev_update_ji, prev_update_kj, prev_update_mk = sgd(a_k_out, a_j_out, x_batch, a_m_out,
                                                                                   y_batch,
                                                                                   w_kj, w_ji, w_mk, lr,
                                                                                   normalization_factor,
                                                                                   prev_update_kj,
                                                                                   prev_update_ji,
                                                                                   prev_update_mk, my)

            if i % check_step == 0:
                STEP.append(iteration)
                a_m_train, a_k_train, a_j_train = forward_pass(w_kj, w_ji, w_mk, x_train)
                a_m_val, a_k_val, a_j_val = forward_pass(w_kj, w_ji, w_mk, x_val)
                a_m_test, a_k_test, a_j_test = forward_pass(w_kj, w_ji, w_mk, x_test)

                TEST_ACC.append(accuracy(y_test, a_m_test))
                VAL_ACC.append(accuracy(y_val, a_m_val))
                TRAIN_ACC.append(accuracy(y_train, a_m_train))

                TRAIN_LOSS.append(cross_entropy_loss(a_m_train, y_train))
                VAL_LOSS.append(cross_entropy_loss(a_m_val, y_val))
                TEST_LOSS.append(cross_entropy_loss(a_m_test, y_test))

                if should_early_stop(VAL_LOSS):
                    print("Early stop at epoch:", epoch)
                    return w_ji, w_kj, w_mk, meta

        lr = learning_rate_annealing(initial_lr, iteration, lr_decay)
        print("\nEpoch", epoch, "Complete:")
        print(
            "Train loss: {:.4f} Test loss: {:.4f} Val loss: {:.4f}".format(TRAIN_LOSS[-1], TEST_LOSS[-1], VAL_LOSS[-1]))

    return w_ji, w_kj, w_mk, meta
