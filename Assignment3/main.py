import pickle
import gzip
import numpy as np


def compute_t(label, dim):
    t = np.zeros(dim)
    t[label] = 1
    return t


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def softmax_activation(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum(axis=1)


def train(training, data_set, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias):
    data = data_set[0]
    labels = data_set[1]
    sigmoid_all_elem = np.vectorize(sigmoid)
    nr_iterations = 0

    while nr_iterations < 5:
        correct_classified = 0
        for idx, input in enumerate(data):
            z = np.dot(input, hidden_layer_weights) + hidden_layer_bias  # (1,784) x (784,100) + (1,100)
            y_hidden = sigmoid_all_elem(z)  # (1, 100)
            z = np.dot(y_hidden, output_layer_weights) + output_layer_bias  # (1,100) x (100,10) + (1,10)
            y_output = softmax_activation(z)  # (1, 10)
            predicted = np.argmax(y_output)
            expected = labels[idx]
            if predicted == expected:
                correct_classified += 1
            if not training:
                break
            t = compute_t(labels[idx], 10)
            output_layer_errors = y_output - t # y_output * (1 - y_output) * (y_output - t)
            # BACKPROPAGATION
            hidden_layer_errors = y_hidden * (1 - y_hidden) * np.dot(output_layer_errors, output_layer_weights.T) # (1, 100)
            # adjust weights
            output_layer_weights = output_layer_weights - (lr * (output_layer_errors.T * y_hidden)).T # (100, 10)
            output_layer_bias = output_layer_bias - (lr * output_layer_errors) # (1,10)
            hidden_layer_weights = hidden_layer_weights - (lr * (hidden_layer_errors.T * input.T)).T # (784, 100)
            hidden_layer_bias = hidden_layer_bias - (lr * hidden_layer_errors) # (1, 100)
        percentage = correct_classified / data.shape[0] * 100
        nr_iterations += 1
        print("iteration:", nr_iterations, "correct classified:", correct_classified,
              "(" + str(percentage) + ")")
        if not training:
            break

    return hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias


with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

lr = np.random.default_rng().uniform(0.1, 0.3)
hidden_layer_weights = np.random.default_rng().uniform(-0.3, 0.3, (784, 100))
output_layer_weights = np.random.default_rng().uniform(-0.3, 0.3, (100, 10))
hidden_layer_bias = np.random.rand(1, 100)
output_layer_bias = np.random.rand(1, 10)

# lr = 0.5
# train_set = (np.array([[2, 6]]), np.array([0]))
# hidden_layer_weights = np.array([[-3, 6], [1, -2]])
# output_layer_weights = np.array([[8], [4]])
# hidden_layer_bias = np.array(0)
# output_layer_bias = np.array(0)

print("Learning rate", lr)
print("TRAINING")
hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias = train(True, train_set, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias)
print("VALIDATION")
hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias = train(True, valid_set, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias)
print("TESTING")
hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias = train(False, test_set, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias)
