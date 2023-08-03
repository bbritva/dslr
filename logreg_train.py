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
def norm_data(x):
    x_max = max(x)
    x_min = min(x)
    return 


@_guard_
def read_data(filename):
    stats = pickle.load(open("stats.pickle", 'rb'))
    try:
        data = pd.read_csv("datasets/" + filename)
    except FileNotFoundError:
        data = pd.read_csv("../datasets/" + filename)
    return data, stats


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
def prepare_target_values(data):
    houses = np.array(data["Hogwarts House"]).reshape((-1, 1))
    Y = np.zeros((houses.shape[0], 5), dtype='int8')
    for i, house in enumerate(houses):
        Y[i][houses_index[house[0]]] = 1
    Y[:,4] = np.array([houses_index[key] for key in houses.flatten()])
    return Y

@_guard_
def normalize_data(X, features, stats):
    for feature in features:
        x_min = stats[feature]['min']
        x_max = stats[feature]['max']
        X[feature] = (X[feature] - x_min) / (x_max - x_min)
    return X


@_guard_
def fill_nan(X, features, stats):
    for key in houses_index:
        for feature in features:
            X[(X["Hogwarts House"] == key) & pd.isna(X[feature])][feature] = stats[feature][key]
    return X
    

@_guard_
def prepare_features(data, stats):
    features = data.columns.values[6:]
    X = fill_nan(data, features, stats)
    X = normalize_data(X, features, stats)
    return np.array(X)


@_guard_
def main(filename):
    data, stats = read_data(filename)
    if data is None or stats is None:
        print("File reading error!")
        exit()
    Y = prepare_target_values(data)
    X = prepare_features(data, stats)
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
