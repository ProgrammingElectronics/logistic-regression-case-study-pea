import unittest
import numpy as np
from logistic_regression import *


class LogisticRegressionTests(unittest.TestCase):

    def test_sigmoid_function(self):
        self.assertEqual(sigmoid(0.0), 0.5)
        input = np.array([0, 0])
        output = np.array([0.5, 0.5])
        self.assertTrue(np.array_equal(sigmoid(input), output))

    def test_linear_regression(self):
        x = np.array([1, 3])
        w = np.array([2, 1])
        b = 5
        self.assertEqual(linear_regression(x, w, b), 10)

    def test_logistic_regression_loss_function(self):
        
        # Value for making prediction sigmoid(w⋅x+b)
        x = np.array([1,3])
        w = np.array([2,1])
        b = -5
        # target value
        target = 1
        
        test_loss = logistic_regression_loss(x,w,b,target)
        actual_loss = -np.log(0.5)

        self.assertEqual(test_loss, actual_loss)
        
    def test_logistic_regression_cost(self):

        # Value for making prediction sigmoid(w⋅x+b) - result for both is 0.5
        X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
        w = np.array([1,1])
        b = -3
        y = np.array([0, 0, 0, 1, 1, 1])
        
        test_cost = 0.36686678640551745

        self.assertEqual(logistic_regression_cost(X,w,b,y),test_cost)


    def test_compute_logistic_gradient_descent(self):
        
        self.assertEqual(logistic_regression_cost(X,w,b,y),test_cost)