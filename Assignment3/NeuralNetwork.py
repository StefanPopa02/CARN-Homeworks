import random

import numpy as np


def softmax_activation(z):
    e_z = np.exp(z)
    return e_z / np.sum(e_z, axis=0)


def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))


class NeuralNetwork:
    def __init__(self, train_set, test_set):
        self.no_of_epochs = 300
        self.test_set_input, self.test_set_label = test_set[0], test_set[1]
        self.train_set_input, self.train_set_label = train_set[0], train_set[1]
        self.convert_in_one_hot_vector(self.train_set_label)
        self.no_data_samples = len(self.train_set_input)
        self.mini_batch_len = 1
        self.learning_rate, self.reg_param, self.momentum = 0.7, 0.1, 0.9
        self.layers_sizes = [len(self.train_set_input[0]), 100, 10]
        self.weights, self.velocity_weights = [], []
        self.biases, self.velocity_biases = [], []
        self.initialize_biases()
        self.initialize_weights()

    @staticmethod
    def get_one_hot_vector(labels):
        res = np.eye(10)[np.array(labels).reshape(-1)]
        return res.reshape(list(labels.shape) + [10])

    def convert_in_one_hot_vector(self, labels):
        self.train_set_label = self.get_one_hot_vector(labels)

    def initialize_weights(self):
        self.weights = [
            np.random.randn(self.layers_sizes[i], self.layers_sizes[i - 1]) / np.sqrt(self.layers_sizes[i - 1]) for i in
            range(1, len(self.layers_sizes))]
        for i in self.weights:
            self.velocity_weights.append(np.zeros_like(i))

    def initialize_biases(self):
        self.biases = [np.random.randn(self.layers_sizes[i], 1) for i in range(1, len(self.layers_sizes))]
        for i in self.biases:
            self.velocity_biases.append(np.zeros_like(i))

    def feed_forward_matrix(self, x):
        activations, net_inputs = [], []
        activation_predecesor = x
        activations.append(activation_predecesor)
        for i in range(len(self.layers_sizes) - 2):
            net_input = np.dot(self.weights[i], activation_predecesor.T) + self.biases[i]
            net_inputs.append(net_input.T)
            activation_predecesor = sigmoid_activation(net_input)
            activations.append(activation_predecesor.T)
        net_input_last = np.dot(self.weights[-1], activation_predecesor) + self.biases[-1]
        net_inputs.append(net_input_last.T)
        activation_last = softmax_activation(net_input_last)
        activations.append(activation_last.T)
        return net_inputs, activations, activation_last

    def feed_forward_one(self, x):
        activation_pred = x.reshape(784, 1)
        for i in range(len(self.layers_sizes) - 2):
            net_input = np.dot(self.weights[i], activation_pred) + self.biases[i]
            activation_pred = sigmoid_activation(net_input)
        net_input_last = np.dot(self.weights[-1], activation_pred) + self.biases[-1]
        activation_last = softmax_activation(net_input_last)
        return activation_last

    @staticmethod
    def cross_entropy_derivative(output, target):
        return output - target

    def backward(self, net_inputs, activations, label):
        changes_w, changes_b = [np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]
        error = self.cross_entropy_derivative(activations[-1], label)
        changes_b[-1], changes_w[-1] = error.sum(axis=0).reshape(error.shape[1], 1), np.dot(error.T, activations[-2])
        for i in range(2, len(self.layers_sizes)):
            sd = self.sigmoid_derivative(net_inputs[-i])
            error = np.dot(self.weights[-i + 1].T, error.T) * sd.T
            changes_b[-i], changes_w[-i] = error.sum(axis=1).reshape(error.shape[0], 1), np.dot(error,
                                                                                                activations[-i - 1])
        return changes_b, changes_w

    @staticmethod
    def sigmoid_derivative(z):
        return sigmoid_activation(z) * (1 - sigmoid_activation(z))

    def get_mini_batch(self):
        mini_batch_index = random.sample(range(0, self.no_data_samples), self.mini_batch_len)
        mini_batch_labels = np.array([self.train_set_label[i] for i in mini_batch_index])
        mini_batch_input = np.array([self.train_set_input[i] for i in mini_batch_index])
        return mini_batch_input, mini_batch_labels

    def get_delta_weight(self, w, nw, v):
        return self.learning_rate * (self.reg_param / self.no_data_samples) * w + (
                self.learning_rate / self.mini_batch_len) * nw + self.momentum * v

    def get_delta_bias(self, v, nb, ):
        return nb * (self.learning_rate / self.mini_batch_len) + self.momentum * v

    def train(self):
        for i in range(self.no_of_epochs):
            mini_batch_input, mini_batch_labels = self.get_mini_batch()
            net_inputs, activations, output = self.feed_forward_matrix(mini_batch_input)
            changes_b, changes_w = self.backward(net_inputs, activations, mini_batch_labels)
            self.velocity_weights = [self.get_delta_weight(w, nw, v) for w, nw, v in
                                     zip(self.weights, changes_w, self.velocity_weights)]
            self.weights = [w - nw for w, nw in zip(self.weights, self.velocity_weights)]
            self.velocity_bias = [self.get_delta_bias(v, nb) for v, nb in zip(self.velocity_biases, changes_b)]
            self.biases = [b - nb for b, nb in zip(self.biases, self.velocity_biases)]
        self.accuracy()

    def accuracy(self):
        self.convert_in_one_hot_vector(self.test_set_label)
        results = [np.argmax(self.feed_forward_one(x)) for x, y in zip(self.test_set_input, self.test_set_label)]
        print(sum(int(x == y) for x, y in zip(results, self.test_set_label)) / len(self.test_set_label) * 100)
