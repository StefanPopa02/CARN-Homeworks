import pickle
import gzip
import numpy as np


def compute_t(label):
    t = np.zeros(10)
    t[label] = 1
    return t


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def feed_forward(data_set, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias):
    data = data_set[0]
    labels = data_set[1]
    sigmoid_all_elem = np.vectorize(sigmoid)
    nr_iterations = 0

    while nr_iterations < 10:
        correct_classified = 0
        for idx, input in enumerate(data):
            z = np.dot(input, hidden_layer_weights) + hidden_layer_bias
            y_hidden = sigmoid_all_elem(z)
            z = np.dot(y_hidden, output_layer_weights) + output_layer_bias
            y_output = sigmoid_all_elem(z)
            predicted = np.argmax(y_output) + 1
            expected = labels[idx]
            if predicted == expected:
                correct_classified += 1
            t = compute_t(labels[idx])
            output_layer_errors = y_output * (1 - y_output) * (y_output - t)
            # BACKPROPAGATION
            hidden_layer_errors = y_hidden * (1 - y_hidden) * np.dot(output_layer_errors, output_layer_weights.T)
            # adjust weights
            output_layer_weights = output_layer_weights - (lr * (output_layer_errors.T * y_hidden)).T
            output_layer_bias = output_layer_bias - (lr * output_layer_errors)
            hidden_layer_weights = hidden_layer_weights - (lr * (hidden_layer_errors.T * input.T)).T
            hidden_layer_bias = hidden_layer_bias - (lr * hidden_layer_errors)
        percentage = correct_classified / data.shape[0] * 100
        nr_iterations += 1
        print("iteration:", nr_iterations, "correct classified:", correct_classified,
              "(" + str(percentage) + ")")


with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

# lr = np.random.default_rng().uniform(0.1, 0.3)
lr = 3.0
print("Learning rate", lr)
hidden_layer_weights = np.random.default_rng().uniform(-0.3, 0.3, (784, 100))
output_layer_weights = np.random.default_rng().uniform(-0.3, 0.3, (100, 10))
hidden_layer_bias = np.random.rand(1, 100)
output_layer_bias = np.random.rand(1, 10)

feed_forward(train_set, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias)
