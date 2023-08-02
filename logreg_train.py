import pandas as pd
import numpy as np
import sys
import pickle

from my_logistic_regression import MyLogisticRegression as MyLR

max_iter = 1e4
alpha = 1e-2
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


def _drop_nan_(func):
    def wrapper(arr, *args, **kwargs):
        arr = arr[~np.isnan(arr)]
        return (func(arr, *args, **kwargs))
    return wrapper


def _to_numpy_(func):
    def wrapper(arr, *args, **kwargs):
        if not isinstance(arr, np.ndarray):
            arr = arr.to_numpy()
        return (func(arr, *args, **kwargs))
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


# @_guard_
def train_model(x_train, y_train):
    theta = np.zeros((14, 1))
    my_lreg = MyLR(theta, alpha=alpha, max_iter=max_iter)
    my_lreg.fit_(x_train, y_train)
    return my_lreg


# @_guard_
def train_models(X, Y):
    models = []
    for i in range(4):
        models.append(train_model(X, Y[:, i].reshape((-1, 1))).theta)
        print(models[-1])
    print("Models trained")
    return models


def fill_nan(X, features):
    with open("medians.pickle", 'rb') as my_file:
        medians = pickle.load(my_file)
        print(medians)

        for key in houses_index:
            for feature in features:
                X[(X["Hogwarts House"] == key) & pd.isna(X[feature])] = medians[feature][key]
        
        return np.array(X[features])


def main(filename):
    data = read_data(filename)
    if data is None:
        print("File reading error!")
        exit()
    # data = data.dropna()
    features = data.columns.values[6:]
    houses = np.array(data["Hogwarts House"]).reshape((-1, 1))
    Y = np.zeros((houses.shape[0], 4), dtype='int8')
    houses_index = {
        "Ravenclaw": 0,
        "Slytherin": 1,
        "Gryffindor": 2,
        "Hufflepuff": 3
    }
    for i, house in enumerate(houses):
        Y[i][houses_index[house[0]]] = 1
    X = fill_nan(data, features)
    print(X)

    models = train_models(X, Y)
    with open("model.pickle", 'wb') as my_file:
        pickle.dump(models, my_file)
        print("All results are saved =)")



if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print("Please, provide the filename in the program arguments")
        exit()
    filename = sys.argv[1]
    main(filename)
