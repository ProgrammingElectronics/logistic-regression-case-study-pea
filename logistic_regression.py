# Function for running logistic regression
import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (n dimensional array): A scalar, numpy array of any size

    Returns:
        (n dimensional array): sigmoid(z), with same shape as z
    """

    return 1 / (1 + np.exp(-z))


def linear_regression(x,w,b):
    """
    compute Fwb(x) = w₁⋅x₁ + w₂⋅x₂ + wₙ⋅xₙ + b 

    Args:
        x (ndarray): Shape (n,) example w/ multiple features
        w (ndarray): Shape (n,) model parameters
        b (scalar): model parameter often called "bias"

    Returns:
        (scalar): prediction
    """
    return np.dot(w,x) + b

def logistic_regression_loss(x,w,b,y):
    """
    compute loss = (-y)(log(Fwb(x)) - actual)² - (1-y)()

    Args:
        x (ndarray): Shape (m,) m features
        w (ndarray): Shape (m,) m model parameters
        b (scalar): model parameter often called "bias"
        y (ndarray): Shape (m,) m target value 
    """
    model = linear_regression(x,w,b)
    pred = sigmoid(model)

    return -y * np.log(pred) - (1-y) * np.log(1 - pred)

def logistic_regression_cost(X,w,b,y):
    """
    compute J(w,b) = 1/m sum(0...m)[-y * log(Fwb(x)) - (1-y) * log(1-Fwb(x))]

    Args:
        x (ndarray): Shape (m,n) m examples, with n features
        w (ndarray): Shape (n,) m model parameters
        b (scalar): model parameter often called "bias"
        y (ndarray): Shape (m,) m target values

    """
    
    m = X.shape[0]

    cost = 0

    for i in range(m):
        loss = logistic_regression_loss(X[i],w,b,y[i])
        cost += loss

    return cost / m