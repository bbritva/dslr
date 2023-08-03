import pandas as pd
import numpy as np
import sys
import pickle

from my_logistic_regression import MyLogisticRegression as MyLR
from data_preparator import DataPreparator as DP   

max_iter = 5e4
alpha = .2

def _guard_(func):
    def wrapper(*args, **kwargs):
        try:
            return (func(*args, **kwargs))
        except Exception as e:
            print(e)
            return None
    return wrapper


@_guard_
def read_data(filename):
    try:
        data = pd.read_csv("datasets/" + filename)
    except FileNotFoundError:
        data = pd.read_csv("../datasets/" + filename)
    return data


@_guard_
def train_model(x_train, y_train):
    theta = np.zeros((x_train.shape[1] + 1, 1))
    my_lreg = MyLR(theta, alpha=alpha, max_iter=max_iter)
    my_lreg.fit_(x_train, y_train)
    return my_lreg


@_guard_
def train_models(X, Y):
    models = []
    for i in range(4):
        models.append(train_model(X, Y[:, i].reshape((-1, 1))).theta)
    print("Models trained")
    return models

@_guard_
def validate_models(X, Y, models):
    thetas = models
    y_hat = []
    for theta in thetas:
        mdl = MyLR(theta, alpha=alpha, max_iter=max_iter)
        y_hat.append(mdl.predict_(X))
    y_hat = np.c_[y_hat[0], y_hat[1], y_hat[2], y_hat[3]]
    y_hat = np.argmax(y_hat, axis=1).reshape((-1, 1))

    """ Output """
    res = y_hat == Y
    print("Correct predictions =", res.sum())
    print("Wrong predictions =", res.shape[0] - res.sum())

   
@_guard_
def main(filename):
    data = read_data(filename)
    if data is None:
        print("File reading error!")
        exit()
    data_preparator = DP()
    Y = data_preparator.prepare_target_values(data)
    X = data_preparator.prepare_features(data)
    train_set, cv_set = data_preparator.split_data(np.c_[X, Y])
    print(train_set.shape, cv_set.shape)
    models = train_models(train_set[:, :-5], train_set[:, -5:-1])
    print(models)
    validate_models(cv_set[:, :-5], cv_set[:, -1:], models)
    with open("model.pickle", 'wb') as my_file:
        pickle.dump(models, my_file)
        print("All results are saved =)")


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print("Please, provide the filename in the program arguments")
        exit()
    filename = sys.argv[1]
    main(filename)
