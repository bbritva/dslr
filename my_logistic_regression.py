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
        y_hat = 1 / (1 + self.eps ** -(x.dot(self.theta)))
        return x.T.dot(y_hat - y) / y.shape[0]

    @_guard_
    def show_loss(self, losses):
        plt.plot(np.arange(len(losses)), losses)
        plt.title("Progress of losses")
        plt.xlabel("Iterations")
        plt.ylabel("Loss value")
        plt.grid()
        plt.show()

   
    @_guard_
    def fit_stochastic(self, x, y, n_cycles=1, batch_size=1):
        self.eps = np.full(1, math.e).reshape((-1, 1))
        self.ones = np.ones(y.shape)
        start = time.time()
        losses = [self.loss_(y, self.predict_(x))]
        print("\r%3d%%, time =%5.2fs" % (0, 0), end="")
        for j in range(n_cycles):
            index = np.random.permutation(x.shape[0])
            x_curr = x[index]
            y_curr = y[index]
            self.x_ = np.c_[np.ones(x.shape[0]), x_curr]
            print(y.shape, y_curr.shape)
            for i in range(0, x.shape[0], batch_size):
                self.theta -= self.alpha * self.gradient(self.x_[i:i + batch_size], y_curr[i:i + batch_size])
                y_hat = 1 / (1 + math.e ** -(self.x_.dot(self.theta)))
                losses.append(self.loss_(y_curr, y_hat))
            now = time.time() - start
            print("\r%3d%%, time = %5.2fs" % ((j + 1), now), end="")
        print("")
        self.show_loss(losses)
        return self.theta
    
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
            for i in range(cycles):
                self.theta -= self.alpha * self.gradient(self.x_, y)
                y_hat = 1 / (1 + math.e ** -(self.x_.dot(self.theta)))
                losses.append(self.loss_(y, y_hat))
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

