import numpy as np


def create_validation_data(x, y, percentage_train):
    items = np.arange(x.shape[0])
    np.random.shuffle(items)
    idx_train = items[:int(percentage_train * x.shape[0])]
    idx_val = items[int(percentage_train * x.shape[0]):]

    x_validation_s = x[idx_val]
    y_validation_s = y[idx_val]
    x_train_s = x[idx_train]
    y_train_s = y[idx_train]
    return x_train_s, y_train_s, x_validation_s, y_validation_s


def one_hot_encode(y, num_classes):
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return np.reshape(encoded, y.shape + (num_classes,))


def bias_trick(x):
    return np.concatenate((x, np.ones((len(x), 1))), axis=1)
