import pickle
import gzip
import numpy as np


def activation(z):
    result = np.array([])
    for input in z.T:
        if input > 0:
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)
    return result


def adjust_weights(weights, t, output, x):
    adjusting_values = np.dot(np.expand_dims(x, axis=1), np.expand_dims(t - output, axis=0)) * lr
    weights += adjusting_values

def compute_t(label):
    t = np.zeros(10)
    t[label] = 1
    return t


def training(train_set, weights, bias):
    train_data = train_set[0]
    train_labels = train_set[1]
    for idx, input in enumerate(train_data):
        z = np.dot(input, weights) + bias
        output = activation(z)
        t = compute_t(train_labels[idx])
        adjust_weights(weights, t, output, input)


with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

lr = np.random.rand(1)
weights = np.random.rand(784, 10)
bias = np.random.rand(1, 10)
training(train_set, weights, bias)

print("Assignment2")
