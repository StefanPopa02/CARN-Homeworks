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


def adjust_weights(weights, t, output, x, bias):
    adjusting_values = np.dot(np.expand_dims(x, axis=1), np.expand_dims(t - output, axis=0)) * lr
    weights += adjusting_values
    bias += (t - output) * lr


def compute_t(label):
    t = np.zeros(10)
    t[label] = 1
    return t


def training(train_set, weights, bias, training):
    train_data = train_set[0]
    train_labels = train_set[1]
    all_classified = False
    nr_iterations = 0
    percentage = 0
    while not all_classified and nr_iterations < 10:
        all_classified = True
        correct_classified = 0
        for idx, input in enumerate(train_data):
            z = np.dot(input, weights) + bias
            output = activation(z)
            t = compute_t(train_labels[idx])
            if training:
                adjust_weights(weights, t, output, input, bias)
            if not np.array_equal(output, t):
                all_classified = False
            else:
                correct_classified += 1
        percentage = correct_classified / train_data.shape[0] * 100
        nr_iterations += 1
        print("iteration:", nr_iterations, "correct classified:", correct_classified,
              "(" + str(percentage) + ")")
        if not training:
            break

    return weights, bias


with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

lr = np.random.default_rng().uniform(0.1, 0.3)
print("Learning rate", lr)
weights = np.random.default_rng().uniform(-0.3, 0.3, (784, 10))
bias = np.random.rand(1, 10)
print("================START TRAINING================")
weights, bias = training(train_set, weights, bias, True)
print("================START VALIDATING================")
weights, bias = training(valid_set, weights, bias, True)
print("================START TESTING================")
weights, bias = training(test_set, weights, bias, False)
