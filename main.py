import numpy as np
from logistic_regression import *

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  # (m,n)
w = np.zeros_like(X[0])
b = 0
y = np.array([0, 0, 0, 1, 1, 1])
lr = 0.1
reg = 0
epochs = 10000
test_gradient = 0
        
compute_logistic_gradient_descent(X, w, b, y,lr,reg, epochs)