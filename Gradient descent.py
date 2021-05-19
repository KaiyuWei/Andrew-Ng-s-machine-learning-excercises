import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('C:\courses\machine learning\linear regression') # Change the working directory to where the file is.
FileName = 'ex1data1.txt'
data = pd.read_csv(FileName, header=None, names=['Population', 'Profit'])
data.head()

data.plot(kind="scatter", x="Population", y="Profit", figsize=(12, 8))  # Visualize and observe the data

# Create a cost function
def computeCost(X, y, theta):
    """
    This function gives out the value of cost function when using the gradient descent
    to look for the value of theta that leads to the minimal cost function value.
    :param X: feature values in the training set
    :param y: response values in the training set
    :param theta: coefficient vector
    :return: coefficient vector theta that leads to the minimal cost function value
    """
    inner = np.power(((X * theta.T) - y), 2)  # The multiplication here is a dot product between matrices.
    return np.sum(inner) / (2 * len(X))

# Insert a column in the data frame
data.insert(0, 'Ones', 1) # Insert a column for which all the values are 1, so that when we make the multiplication
# later, we always use 1 to mutiply with theta_0.
# Set X and y
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]  # X is the first two columns in data
y = data.iloc[:, cols - 1:cols]  # y is the last column in data

# Transfer all input arguments to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
# Initialize thita
theta = np.matrix(np.array([0, 0]))  # We have one feature here so we need one coefficient and on intercept.

# Calculate the current value of cost function
computeCost(X, y, theta)

# Define the gradient decent function
def gradientDescent(X, y, theta, alpha, iters):
    """
    This function performs the gradient descent method to look for theta values to minimize
    the value of the cost function.
    :param X: feature values in training set
    :param y: response values in training set
    :param theta: np array, current coefficient and intercept values
    :param alpha: learning rate.
    :param iters: number of iterations we want to try for different values of theta
    :return: optimal coefficient values and cost function values
    """
    temp = np.matrix(np.zeros(theta.shape))  # The same shape as the theta matrix
    parameters = int(theta.ravel().shape[1])  # The number of parameters in theta
    cost = np.zeros(iters)  # Array in order to record different value of cost function with different parameters

    for i in range(iters):
        error = (X * theta.T) - y  # X * theta.T is the values of the hypothesis function.
        for j in range(parameters):  # The number of items in array theta
            term = np.multiply(error, X[:, j])  # Element-wise multiplication. When j = 0, we always use 1 to
            # multiply with error, which complies with the formula when theta is theta_0
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))  # Updata of values of theta
            theta = temp
            cost[i] = computeCost(X, y, theta)

    return theta, cost


# Initialize the parameters
alpha = 0.01
iters = 1000

g, cost = gradientDescent(X, y, theta, alpha, iters)  # Get values of theta and cost

# visualize the values to test if the theta fit the data
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)  # Pay attention to the way to access entries in a matrix. g[0,0], g[0,1] not g[0], g[1]

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'g', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs Population Size')
plt.show()

# Visualize the process of the gradient descent
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


# Optional excercise: linear regression with multiple variables.
FileName = 'ex1data2.txt'
data2 = pd.read_csv(FileName, header=None, names=['Size', 'Bedrooms', 'Price'])
data2.head()

data2 = (data2 - data2.mean()) / data2.std() # Feature-scaling of the data element-wise

data2.insert(0, 'Ones', 1)  # Add ones column, for computation by matrix in following code

# Set X and y
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]

# Convert X, y to matrices
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array(np.zeros(3)))
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# Get the value of the cost function
computeCost(X2, y2, g2)

# Visualize the training process
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Errors vs. Training Epoch')
plt.show()

# Use normal equation to solve the multi-variable problem
def normalEqn(X, y):
    """
    returns the value of theta that minimize the cost function
    :param X: values of feature(s), including X_0 = 1
    :param y: values of response
    :return: values of theta that minimize the cost function
    """
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # @ is the dot product operator
    return theta

final_theta2 = normalEqn(X, y)
