import numpy as np
import time
import math
import matplotlib.pyplot as plt



def _guard_(func):
    def wrapper(*args, **kwargs):
        try:
            return (func(*args, **kwargs))
        except Exception as e:
            print(func.__name__ + ': ', e)
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

    @_guard_
    def gradient(self, x, y):
        y_hat = 1 / (1 + self.eps ** -(self.x_.dot(self.theta)))
        return self.x_.T.dot(y_hat - y) / y.shape[0]

    @_guard_
    def stochastic_gradient(self, x, y):
        y_hat = 1 / (1 + self.eps ** -(self.x_.dot(self.theta)))
        return self.x_.T.dot(y_hat - y) / y.shape[0]

    @_guard_
    def show_loss(self, losses):
        plt.plot(np.arange(len(losses)), losses)
        plt.title("Progress of losses")
        plt.xlabel("Iterations")
        plt.ylabel("Loss value")
        plt.grid()
        plt.show()

    @_guard_
    def grad_desc_fit(self, cycles, x, y):
        losses = []
        for i in range(cycles):
            self.theta -= self.alpha * self.gradient(x, y)
            losses.append(self.loss_(y, self.predict_(x)))
        return losses
    
    @_guard_
    def stochastic_grad_desc_fit(self, cycles, x, y):
        losses = []
        for i in range(cycles):
            self.theta -= self.alpha * self.stochastic_gradient(x, y)
            losses.append(self.loss_(y, self.predict_(x)))
        return losses
    
    @_guard_
    def fit_(self, x, y, isStochastic = False):
        self.eps = np.full(y.shape, math.e)
        self.x_ = np.c_[np.ones(x.shape[0]), x]
        self.ones = np.ones(y.shape)
        start = time.time()
        cycles = int(self.max_iter / 100)
        losses = [self.loss_(y, self.predict_(x))]
        print("\r%3d%%, time =%5.2fs" % (0, 0), end="")
        for j in range(100):
            if isStochastic :
                losses += self.grad_desc_fit(cycles, x, y)
            else:
                losses += self.stochastic_grad_desc_fit(cycles, x, y)
            now = time.time() - start
            print("\r%3d%%, time = %5.2fs" % ((j + 1), now), end="")
        print("")
        self.show_loss(losses)
        return self.theta
    
    @_guard_
    def loss_(self, y, y_hat, eps=1e-15):
        m1 = (y.T.dot(np.log(y_hat + eps)))
        m2 = (self.ones - y).T.dot(np.log(self.ones - y_hat + eps))
        return float((m1 + m2)) / (-y.shape[0])

    @_guard_
    def loss_elem_(self, y, y_hat):
        return (y - y_hat) ** 2

    @staticmethod
    @_guard_
    def mse_(y, y_hat):
        return float((y_hat - y).T.dot((y_hat - y))) / (y.shape[0])

