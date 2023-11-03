# Function for running logistic regression
import numpy as np
import math

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
    compute loss = -y * log(Fwb(x)) - (1-y) * log(1 - Fwb(x))

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
    -> where m is the number of examples in the training set

    Args:
        x (ndarray): Shape (m,n) m examples, with n features
        w (ndarray): Shape (n,) m model parameters
        b (scalar): model parameter often called "bias"
        y (ndarray): Shape (m,) m target values

    """
    
    m = X.shape[0] # number of training examples

    cost = 0

    for i in range(m):
        loss = logistic_regression_loss(X[i],w,b,y[i])
        cost += loss

    return cost / m

def compute_logistic_gradient_descent(X,w,b,y,alpha,lambda_,epochs):
    """
    Compute parameters for logistic regression model using gradient descent

    For each epoch:
        -Compute cost: J(w,b) = 1/m sum(0...m)[-y * log(Fwb(x)) - (1-y) * log(1-Fwb(x))]
        -Update parameters based on batch gradient descent

    For each training example:
        - Compute prediction sigmoid(Fwb(X)) = 1/1+e^(-wx+b)
        - computer error:  prediction - actual
        - compute gradient update
            ↳ w = w - ⍺ (prediction - actual)*X  
            ↳ b = b - ⍺ (prediction - actual)
    
    Args:
        X (ndarray): Shape (m,n) m examples, with n features
        w (ndarray): Shape (n,) m model parameters
        b (scalar): model parameter often called "bias"
        y (ndarray): Shape (m,) m target values
        alpha (scalar): learning rate ⍺
        lambda_ (scalar): lambda regularization parameter
        epochs (scalar): iterations of gradient descent to perform 
    """

    m = X.shape[0] # number of training examples
    n = X.shape[1] # number of features

    for j in range(epochs):

        dj_dw_temp = np.zeros(n) 
        dj_db_temp = 0

        # Compute gradient            
        for i in range(m):
            z = linear_regression(X[i],w,b)
            
            pred = sigmoid(z)
            error = pred - y[i]

            # gradients
            dj_dw_temp += np.dot(error, X[i])
            dj_db_temp += error
        
        w -= alpha * dj_dw_temp/m
        b -= alpha * dj_db_temp/m

        # Compute cost
        cost = logistic_regression_cost(X,w,b,y)
        if j % math.ceil(epochs / 10) == 0:
            print(f"epoch:{j} Cost = {cost}")

    return cost