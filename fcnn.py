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


def cross_entropy_loss(output, targets):
    assert output.shape == targets.shape
    # output[output == 0] = 1e-8
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    # print(cross_entropy.shape)
    return cross_entropy.mean()

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