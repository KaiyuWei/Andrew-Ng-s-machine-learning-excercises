import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  # For get and change the working directory


fileName = 'ex2data1.txt'
os.chdir('C:\courses\machine learning\logistic regression')
data = pd.read_csv(fileName, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()

# Create a scatter plot and use different colors to indicate different results
positive = data[data['Admitted'].isin([1])]  # Select rows in which the result is 1
negative = data[data['Admitted'].isin([0])]  # Select rows in which the result is 0.
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')  # Scatter plot for admitted students
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')  # Scatter plot for not admitted students.
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 score')
plt.show()

# Sigmoid function for logistic regression
def sigmoid(z):
    """
    A function that gives the result of the sigmoid function.
    :param z: Numeric number or an array-like object, X * theta.T
    :return: the result of the sigmoid function, numeric or array-like objects
    """
    return 1 / (1 + np.exp(-z))  # np.exp can be operated on array, while math.exp cannot.

# Visualize the function to check if it can work normally
nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()

# Define the cost function
def cost(theta, X, y):
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
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))  # np.multiply is element-wise. First part of the cost function
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))  # Second part of the cost function
    return np.sum(first - second) / len(X)

# In order to perform the matrix operations, we need to add a ones column in X
data.insert(0, 'Ones', 1)
cols = data.shape[1]  # Get the number of columns in data
X = data.iloc[:, 0:cols-1]  # X is all columns except the last one in data
y = data.iloc[:, cols-1:cols]  # y is the last column
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)
# After create theses arrays we also need to check their shape parameter to make sure that we can run them normally.

cost(theta, X, y)  # The result of current cost function is 0.6931471805599453, which is close to 0.

# Gradient descent function
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)  # Convert arrays to matrices for convenience.

    parameters = int(theta.ravel().shape[1])  # Get the number of parameters
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y  # Generate an array listing all pairs of errors

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad  # Return one step of the gradient descent. Since we are going to use opt.fmin_tnc to find out the
                        # minimal value of the cost function, which need an argument of generating gradient, so we
                        # only generate gradient here.


gradient(theta, X, y)  # check the current step size after one iteration.

import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))  # x0 is the initial value of parameters.
result

# Create a function that use the result of sigmoid function to predict the result of logistic regression.
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# Now use the result from fmin_tnc function to predict the result
theta_min = np.matrix(result[0])  # The first item in tuple "result" is the array of theta
predictions = predict(theta_min, X)  # Get the predict result
correct = [1 if (a == b) else 0 for (a, b) in zip(predictions, y)]  # Get the number of correct predictions
accuracy = (sum(map(int, correct)) % len(correct))  # The remainder is the number of correct predictions
print('accuracy = {0}%'.format(accuracy))


# Regularization
fileName = 'ex2data2.txt'  # File name.
data2 = pd.read_csv(fileName, header=None, names=['Test 1', 'Test 2', 'Accepted'])  # Read data from file.

# Visualize the data
positive = data2[data2.Accepted.isin([1])]
negative = data2[data2.Accepted.isin([0])]  # Split the data into two parts

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='r', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='b', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()

# Create the polynomial feature for these data
degree = 5  # Power parameter constraint that is used to create the polynomial.
x1 = data2['Test 1']  # Store data
x2 = data2['Test 2']  # Store data
data2.insert(3, 'Ones', 1)  # Insert ones in the last column of the data

for i in range(1, degree):
    for j in range(0, i):  # The power of x2
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)  # Power of x1 and x2: 10, 20, 11, 30, 21, 12, 40, 31, 22, 13

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)
data2.head


# Create the cost function with regularized terms
def RegCost(theta, X, y, learningRate):
    """
    Calculating the cost value with regularization
    :param theta:  Coefficient parameters
    :param X: feature values in training set
    :param y: response values in training set
    :param learningRate: parameter alpha for "punishing" coefficients (regularization).
    :return:  the value of the regularized cost function with current theta values.
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)  # Convert input data into matrices for convenient operations

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))  # Theta_0 is not regularized
    return np.sum(first - second) / len(X) + reg


def gradientReg(theta, X, y, learningRate):
    """
    This function executes gradient descent for once.
    :param theta:  Coefficient parameters
    :param X: feature values in training set
    :param y: response values in training set
    :param learningRate: parameter alpha for "punishing" coefficients (regularization).
    :return: updated theta after once gradient descent
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])  # Number of coefficient parameters.
    grad = np.zeros(parameters)   # A array for storing the updated parameters.
    error = sigmoid(X * theta.T) - y  # Difference between predictions and real y values

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if i == 0:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = np.sum(term) / len(X) + learningRate * theta[:, i] / len(X)

    return grad


cols = data2.shape[1]
X2 = data2.iloc[:, 1:cols]
y2 = data2.iloc[:, 0:1]
# Convert them to numpy arrays and initialize the parameters array theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(X2.shape[1])  # The "Ones" column has been added to the data frame.
learningRate = 1  # Initialize the learning rate.

# Check the cost value with current theta
RegCost(theta2, X2, y2, learningRate)
# Run the regularized gradient descent function
gradientReg(theta2, X2, y2, learningRate)

# Use the same function to find the minimal value
result2 = opt.fmin_tnc(func=RegCost, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
result2

theta2_min = np.matrix(result2[0])  # The first item of result2 is the coefficient array
predictions2 = predict(theta2_min, X2)
correct2 = [1 if (a == b) else 0 for (a, b) in zip(predictions2, y2)]
accuracy2 = sum(map(int, correct2)) % len(correct2)
print("accuracy= = {0}%".format(accuracy2))

# Use scikit learn to perform logistic regression
from sklearn import linear_model
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X2, y2.ravel())
model.score(X2, y2)
