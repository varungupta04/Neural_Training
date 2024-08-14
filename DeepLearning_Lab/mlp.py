import numpy as np

class Mlp():
    def __init__(self, size_layers, act_funct='sigmoid', reg_lambda=0, bias_flag=True):
        self.size_layers = size_layers
        self.n_layers = len(size_layers)
        self.act_f = act_funct
        self.lambda_r = reg_lambda
        self.bias_flag = bias_flag
        self.initialize_theta_weights()

    def train(self, X, Y, iterations=400, reset=False):
        n_examples = Y.shape[0]
        if reset:
            self.initialize_theta_weights()
        for iteration in range(iterations):
            self.gradients = self.backpropagation(X, Y)
            self.gradients_vector = self.unroll_weights(self.gradients)
            self.theta_vector = self.unroll_weights(self.theta_weights)
            self.theta_vector = self.theta_vector - self.gradients_vector
            self.theta_weights = self.roll_weights(self.theta_vector)

    def predict(self, X):
        A, Z = self.feedforward(X)
        Y_hat = A[-1]
        return Y_hat

    def initialize_theta_weights(self):
        self.theta_weights = []
        size_next_layers = self.size_layers[1:]
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            if self.act_f == 'sigmoid':
                epsilon = 4.0 * np.sqrt(6) / np.sqrt(size_layer + size_next_layer)
                theta_tmp = epsilon * (np.random.rand(size_next_layer, size_layer + 1) * 2.0 - 1) if self.bias_flag else epsilon * (np.random.rand(size_next_layer, size_layer) * 2.0 - 1)
            elif self.act_f == 'relu':
                epsilon = np.sqrt(2.0 / (size_layer * size_next_layer))
                theta_tmp = epsilon * np.random.randn(size_next_layer, size_layer + 1) if self.bias_flag else epsilon * np.random.randn(size_next_layer, size_layer)
            self.theta_weights.append(theta_tmp)

    def backpropagation(self, X, Y):
        g_dz = self.sigmoid_derivative if self.act_f == 'sigmoid' else self.relu_derivative
        n_examples = X.shape[0]
        A, Z = self.feedforward(X)
        deltas = [None] * self.n_layers
        deltas[-1] = A[-1] - Y
        for ix_layer in np.arange(self.n_layers - 2, 0, -1):
            theta_tmp = self.theta_weights[ix_layer]
            if self.bias_flag:
                theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
            deltas[ix_layer] = (np.matmul(theta_tmp.transpose(), deltas[ix_layer + 1].transpose())).transpose() * g_dz(Z[ix_layer])

        gradients = [None] * (self.n_layers - 1)
        for ix_layer in range(self.n_layers - 1):
            grads_tmp = np.matmul(deltas[ix_layer + 1].transpose(), A[ix_layer]) / n_examples
            if self.bias_flag:
                grads_tmp[:, 1:] += (self.lambda_r / n_examples) * self.theta_weights[ix_layer][:, 1:]
            else:
                grads_tmp += (self.lambda_r / n_examples) * self.theta_weights[ix_layer]
            gradients[ix_layer] = grads_tmp
        return gradients

    def feedforward(self, X):
        g = self.sigmoid if self.act_f == 'sigmoid' else self.relu
        A = [None] * self.n_layers
        Z = [None] * self.n_layers
        input_layer = X

        for ix_layer in range(self.n_layers - 1):
            n_examples = input_layer.shape[0]
            if self.bias_flag:
                input_layer = np.concatenate((np.ones([n_examples, 1]), input_layer), axis=1)
            A[ix_layer] = input_layer
            Z[ix_layer + 1] = np.matmul(input_layer, self.theta_weights[ix_layer].transpose())
            input_layer = g(Z[ix_layer + 1])

        A[self.n_layers - 1] = input_layer
        return A, Z

    def unroll_weights(self, rolled_data):
        unrolled_array = np.concatenate([layer.flatten("F") for layer in rolled_data])
        return unrolled_array

    def roll_weights(self, unrolled_data):
        size_next_layers = self.size_layers[1:]
        rolled_list = []
        extra_item = 1 if self.bias_flag else 0
        for size_layer, size_next_layer in zip(self.size_layers, size_next_layers):
            n_weights = size_next_layer * (size_layer + extra_item)
            data_tmp = unrolled_data[:n_weights].reshape(size_next_layer, size_layer + extra_item, order='F')
            rolled_list.append(data_tmp)
            unrolled_data = np.delete(unrolled_data, np.s_[0:n_weights])
        return rolled_list

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def relu(self, z):
        return np.maximum(z, 0)

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def relu_derivative(self, z):
        return (z > 0).astype(float)
