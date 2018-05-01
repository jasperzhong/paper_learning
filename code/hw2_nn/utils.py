import numpy as np


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def softmax(x):
    x = np.exp(x)
    return x / sum(x)


def relu(x):
    return max(x, 0)


def cross_entropy_loss(pred, label):
    return np.sum(-label * np.log(pred))


def mse_loss(pred, label):
    return np.sum((pred - label) ** 2)