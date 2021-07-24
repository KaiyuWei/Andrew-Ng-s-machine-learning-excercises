import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)  # convert y to one_hot_code format

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# the forward propagation function
def forward_propagate(X, theta1, theta2):
    """
    This function gives out the theta active unit values for each layer.
    :param X: training set without biased units
    :param theta1: theta for the first layer (input layer), 25 x 401 in the example
    :param theta2: theta for the second layer (hidden layer), 10 x 26 in the example
    :return:
    """
    m = X.shape[0]  # 5000 in the example
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)  # add biased units, 5000 x 401 in the example
    z2 = a1 * theta1.T  # 5000 x 25
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)  # add biased units
    z3 = a2 * theta2.T  # 5000 x 10 in the example
    h = sigmoid(z3)
    return a1, z2, a2, z3, h  # a1: 5000 x 401, z2:5000 x 25, a2: 5000 x 26, z3: 5000 x 10, h: 5000 x 10


# Initialize the parameters
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

# random parameter = rand(total number of theta1 and theta2) * (2*eps) – eps
params = np.random.rand(hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) * 0.5 - 0.25
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
# Check the initialized values
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
# cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)


# regularization of the cost function
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    """
    regularized cost function.
    :param params: initialized theta values, in the form of a 1d array
    :param input_size: no. of input features
    :param hidden_size: no. of active units in the hidden layer.
    :param num_labels: no. of classes
    :param X: features in the training set
    :param y: results in the training set
    :param learning_rate:
    :return: value for current params(theta) of regularized cost function
    """
    m = X.shape[0]  # no. of samples
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape params
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # get parameter values from the forward propagate function
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # parts of the cost function
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply(1 - y[i, :], np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m

    # regularize the cost function
    reg_item = float(learning_rate) / (2 * m) * (np.sum(np.power(theta1[:, 1:], 2)) +
                                                 np.sum(np.power(theta2[:, 1:], 2)))  # ignore the biased unit
    J += reg_item
    return J


# check current cost function value with the initialized parameters.
cost(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

# the differentiate of the sigmoid function
def sigmoid_gradient(z):
    """
    the function result is used for calculating the error from the other layer in back propagation.
    :param z: input value of the sigmoid function
    :return: the differentiate of the sigmoid function
    """
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# back propagation function
def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    """
    back propagation function.
    :param params: initialized parameters
    :param input_size: no. of input features.
    :param hidden_size: no. of active units in the hidden layer.
    :param num_labels: no. of output classes.
    :param X: data of features in the training set.
    :param y: data of results in the training set.
    :param learning_rate:
    :return: gradient for the neural network gradient function.
    """
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # extract parameters from "params"
    J = 0
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # get parameter values from the forward propagate function
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply(1 - y[i, :], np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m

    # regularize the cost function
    reg_item = float(learning_rate) / (2 * m) * (np.sum(np.power(theta1[:, 1:], 2)) +
                                                 np.sum(np.power(theta2[:, 1:], 2)))  # ignore the biased unit
    J += reg_item

    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    # back propagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        # get the error of each layer
        d3t = ht - yt  # (1, 10)
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((d3t * theta2), sigmoid_gradient(z2t))  # (1, 26)

        # accumulative big delta
        delta1 = delta1 + d2t[:, 1:].T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the regularization term
    delta1[:, 1:] = delta1[:, 1:] + learning_rate * theta1[:, 1:] / m
    delta2[:, 1:] = delta2[:, 1:] + learning_rate * theta2[:, 1:] / m

    # ravel the gradient matrices to a 1-d array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


# check current gradient for the current initialized parameters
J, grad = backprop(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)

# use the gradient to optimize our prediction
from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})


# use the forward propagation
X = np.matrix(X)
theta1 = np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1)))
theta2 = np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1)))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

# check the accuracy of our prediction
correct = [1 if a == b else 0 for (a, b) in zip(y, y_pred)]
accuracy = np.sum(correct) / len(correct)
print('accuracy = {}%'.format(accuracy * 100))  # accuracy = 99.28%
