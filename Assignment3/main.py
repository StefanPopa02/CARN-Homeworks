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


def weights_changed_output(output_layer_errors, y_hidden):
    weights_changed = []
    for i in range(minibatch_size):
        weights_changed.append(output_layer_errors[i] * y_hidden[i])
    return weights_changed


def feed_forward(data_set, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias,
                 minibatch_size):
    data = data_set[0]
    data_labels = data_set[1]
    sigmoid_all_elem = np.vectorize(sigmoid)
    nr_iterations = 0
    start = 0
    end = minibatch_size

    while nr_iterations < 10:
        correct_classified = 0
        while end < data.shape[0]:
            input = np.array([i for i in data[start:end]])  # (10, 784)
            labels = np.array([i for i in data_labels[start:end]])  # (10, 1)
            z = np.dot(input, hidden_layer_weights) + hidden_layer_bias  # (10,784) x (784,100) + (10,100)
            y_hidden = sigmoid_all_elem(z)  # (10, 100)
            z = np.dot(y_hidden, output_layer_weights) + output_layer_bias  # (10,100) x (100,10) + (10,10)
            y_output = softmax_activation(z)  # (10, 10)
            predicted = [np.argmax(i) for i in y_output]
            expected = labels
            correct_classified += (predicted == expected).sum()  # array of true/false for each element if predicted[i] == expected[i]
            t = np.array([compute_t(i, 10) for i in labels])
            output_layer_errors = y_output - t  # y_output * (1 - y_output) * (y_output - t)
            # BACKPROPAGATION
            hidden_layer_errors = y_hidden * (1 - y_hidden) * np.dot(output_layer_errors, output_layer_weights.T)  # (10, 100)
            # adjust weights
            output_layer_weights = output_layer_weights - (lr * output_layer_errors.reshape(-1, 1).T * y_hidden).T
            # output_layer_weights = output_layer_weights - (lr * output_layer_errors.reshape(-1, 1).T * y_hidden).T  # (100, 10)
            output_layer_bias = output_layer_bias - (lr * output_layer_errors)  # (10 ,10)
            hidden_layer_weights = hidden_layer_weights - (lr * np.sum([hidden_layer_errors[i] * input[i].reshape(-1, 1) for i in range(minibatch_size)], axis=0))
            hidden_layer_bias = hidden_layer_bias - (lr * hidden_layer_errors)  # (1, 100)
            start = end
            end += minibatch_size
        start = 0
        end = minibatch_size
        percentage = correct_classified / data.shape[0] * 100
        nr_iterations += 1
        print("iteration:", nr_iterations, "correct classified:", correct_classified,
              "(" + str(percentage) + ")")


with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

lr = np.random.default_rng().uniform(0.1, 0.3)
minibatch_size = 10
print("Learning rate", lr)
hidden_layer_weights = np.random.default_rng().uniform(-0.3, 0.3, (784, 100))
output_layer_weights = np.random.default_rng().uniform(-0.3, 0.3, (100, 10))
hidden_layer_bias = np.random.rand(minibatch_size, 100)
output_layer_bias = np.random.rand(minibatch_size, 10)

# train_set = (np.array([[2, 6]]), np.array([0]))
# hidden_layer_weights = np.array([[-3, 6], [1, -2]])
# output_layer_weights = np.array([[8], [4]])
# hidden_layer_bias = np.array(0)
# output_layer_bias = np.array(0)

feed_forward(train_set, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias,
             minibatch_size)
