import pandas as pd
import numpy as np
import sys
import pickle

from my_logistic_regression import MyLogisticRegression as MyLR

max_iter = 5e5
alpha = 1e-1
houses_index = {
    "Ravenclaw": 0,
    "Slytherin": 1,
    "Gryffindor": 2,
    "Hufflepuff": 3
}

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
    data = None
    try:
        data = pd.read_csv("datasets/" + filename)
    except FileNotFoundError:
        try:
            data = pd.read_csv("../datasets/" + filename)
        except FileNotFoundError:
            return None
    return data


@_guard_
def train_model(x_train, y_train):
    theta = np.zeros((14, 1))
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
def fill_nan(X, features):
    with open("medians.pickle", 'rb') as my_file:
        medians = pickle.load(my_file)
        for key in houses_index:
            for feature in features:
                X[(X["Hogwarts House"] == key) & pd.isna(X[feature])] = medians[feature][key]
        return np.array(X[features])


@_guard_
def split_data(x):
    np.random.shuffle(x)
    limit_train = int(0.8 * x.shape[0])
    return x[:limit_train,:], x[limit_train:,:]

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
    features = data.columns.values[6:]
    houses = np.array(data["Hogwarts House"]).reshape((-1, 1))
    Y = np.zeros((houses.shape[0], 5), dtype='int8')
    for i, house in enumerate(houses):
        Y[i][houses_index[house[0]]] = 1
    
    print(Y.shape, houses.shape)
    coded_houses = np.array([houses_index[key] for key in houses.flatten()])
    Y[:,4] = coded_houses
    print(Y)
    X = fill_nan(data, features)
    train_set, cv_set= split_data(np.c_[X, Y])
    models = train_models(train_set[:, :-5], train_set[:, -5:-1])

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
