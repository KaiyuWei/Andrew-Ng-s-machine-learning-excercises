import os
import numpy as np
import pandas as pd
import matplotlib as plt
from scipy.io import loadmat

os.chdir('C:\courses\machine learning\week4 neural network')
data = loadmat('ex3data1.mat')
data['X'].shape  # (5000, 400), 5000 training samples, each of which is a 20x20 pixel^2 grey scale image. Each value in X is a greyscale intensity
data['y'].shape  # (5000, 1)

# Sigmoid function
def sigmoid(z):
    """
    A function that gives the result of the sigmoid function.
    :param z: Numeric number or an array-like object, X * theta.T
    :return: the result of the sigmoid function, numeric or array-like objects
    """
    return 1 / (1 + np.exp(-z))


# Cost function
def cost(theta, X, y, learningRate):
    """
    Returns the value of the cost function, given X, y, theta in the training set.
    :param theta: coefficient array
    :param X: feature values in the training set
    :param y: response values in the training set
    :return: the value of the cost function
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))  # Element-wise multiplication
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))  # Element-wise multiplication
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradient(theta, X, y, learningRate):
    """
    Vectorized gradient function, from the input layer to output layer.
    :param theta: Current coefficient values
    :param X: feature values in the training set
    :param y: response values in the training set
    :param learningRate: regularization parameter.
    :return: gradient values for current coefficient theta
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])  # NUmber of parameters in theta
    error = sigmoid(X * theta.T) - y
    grad = ((X.T * error) / len(X)).T + learningRate / len(X) * theta
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)  # Intercept gradient is not regularized
    return np.array(grad).ravel()


from scipy.optimize import minimize
def one_vs_all(X, y, num_labels, learning_rate):
    """

    :param X: feature values in the training set
    :param y: prediction values in the training set
    :param num_labels: number of numeric labels that need to be predicted
    :param learning_rate: regularization punishment parameter
    :return:
    """
    rows = X.shape[0]
    params = X.shape[1]  # Number of features.
    # k x (n+1) array for the parameters of each of the k classifiers.
    all_theta = np.zeros((num_labels, params + 1))  # Add an extra parameter for the intercept.
    # Insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):  # An biased unit is added, so start from 1
        theta = np.zeros(params + 1)  # initial values for theta are 0's.
        y_i = np.array([1 if label == i else 0 for label in y])  # for one-to-all algorithm. if the label is the current value.
        y_i = np.reshape(y_i, (rows, 1))
        # Minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1, :] = fmin.x

    return all_theta


# In order to figure out the dimension of each array, we use the following code to show the dimension numbers.
rows = data['X'].shape[0]
params = data['X'].shape[1]
all_theta = np.zeros((10, params+1))
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
theta = np.zeros(params + 1)
y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))
X.shape, y_0.shape, theta.shape, all_theta.shape
# Check how many unique values in labels.
np.unique(data['y'])

# Run the on-vs-all function
all_theta = one_vs_all(data['X'], data['y'], 10, 1)

# Obtain the probability of each class
def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    # Insert one row to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # Convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    # compute the class probabilities for each class
    h = sigmoid(X * all_theta.T)
    h_argmax = np.argmax(h, axis=1)  # Obtain the indices of the location for the max probability in each row
    h_argmax = h_argmax + 1 # Array in python is 0-indexed.

    return h_argmax



y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct))) / float(len(correct))
accuracy
