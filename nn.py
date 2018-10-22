import pickle

import numpy as np
from matplotlib import pyplot as plt


class Neural_Network(object):
    def __init__(self, input_size, hidden_size, output_size):
        # parameters
        self.inputSize = input_size
        self.hiddenSize = hidden_size
        self.outputSize = output_size

        self.eta = 0.001

        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1) weight matrix from hidden to output layer

    def forward(self, X):
        # forward propagation through our network
        self.z = np.dot(X, self.W1)  # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propagate through the network
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)  # z2 error: how much our hidden layer weights to output
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta * self.eta)  # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta * self.eta)  # adjusting second set (hidden --> output) weights

    def train(self, train_X, train_y, valx, valy, epochs=50):
        train_errs = []
        val_errs = []
        for e in range(epochs):
            print(e)

            o = self.forward(train_X)
            train_errs.append(self.err(o, train_y))
            val_errs.append(self.err(self.forward(valx), valy))

            self.backward(train_X, train_y, o)

        return train_errs, val_errs

    def err(self, o, exp):
        diff = np.argmax(o) - np.argmax(exp)
        return np.sum(diff != 0) / diff.size


def one_hot(data):
    m = []
    for j in data:
        e = np.zeros((10,))
        e[j] = 1.0
        m.append(e)
    return np.array(m)


NN = Neural_Network(784, 20, 10)

f = open('mnist.pkl', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
f.close()

train_size = 10000
val_size = 3000

tr_x, tr_y = training_data[0][:train_size], one_hot(training_data[1][:train_size])
val_x, val_y = training_data[0][train_size:train_size + val_size], one_hot(
    training_data[1][train_size:train_size + val_size])

train_errs, val_errs = NN.train(tr_x, tr_y, val_x, val_y)

plt.plot(train_errs, label='train error')
plt.plot(val_errs, label='valid error')

plt.show()
