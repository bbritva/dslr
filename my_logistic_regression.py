import numpy as np
import time
import math


def _guard_(func):
    def wrapper(*args, **kwargs):
        try:
            return (func(*args, **kwargs))
        except Exception as e:
            print(func.__name__ + ': ' + e)
            return None
    return wrapper


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    @_guard_
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = int(max_iter)
        self.theta = np.array(theta)
        self.theta_ = np.zeros(theta.shape)

        self.eps = np.full(self.theta.shape, math.e)
        self.x_ = np.ones(self.theta.shape)

    @_guard_
    def predict_(self, x):
        x_ = np.c_[np.ones((x.shape[0])), x]
        return 1 / (1 + math.e ** -(x_.dot(self.theta)))

    # @_guard_
    def gradient(self, x, y):
        y_hat = 1 / (1 + self.eps ** -(self.x_.dot(self.theta)))
        return self.x_.T.dot(y_hat - y) / y.shape[0]

    # @_guard_
    def fit_(self, x, y):
        self.eps = np.full(y.shape, math.e)
        self.x_ = np.c_[np.ones(x.shape[0]), x]
        start = time.time()
        cycles = int(self.max_iter / 20)
        print("\r%3d%%, time =%5.2fs" % (0, 0), end="")
        for j in range(20):
            for i in range(cycles):
                self.theta -= self.alpha * self.gradient(x, y)
            now = time.time() - start
            print("\r%3d%%, time = %5.2fs" % ((j + 1) * 5, now), end="")
        print("")
        return self.theta

    @_guard_
    def loss_(self, y, y_hat, eps=1e-15):
        ones = np.ones(y.shape)
        m1 = (y.T.dot(np.log(y_hat + eps)))
        m2 = (ones - y).T.dot(np.log(ones - y_hat + eps))
        return float((m1 + m2)) / (-y.shape[0])

    @_guard_
    def loss_elem_(self, y, y_hat):
        return (y - y_hat) ** 2

    @staticmethod
    @_guard_
    def mse_(y, y_hat):
        return float((y_hat - y).T.dot((y_hat - y))) / (y.shape[0])

