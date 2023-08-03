import pandas as pd
import numpy as np
import sys
import pickle

from my_logistic_regression import MyLogisticRegression as MyLR

max_iter = 1e4
alpha = 1e-2
houses_index = [
    "Ravenclaw",
    "Slytherin",
    "Gryffindor",
    "Hufflepuff"
]

def _guard_(func):
    def wrapper(*args, **kwargs):
        # try:
            return (func(*args, **kwargs))
        # except Exception as e:
        #     print(e)
        #     return None
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
def fill_nan(X, features):
    with open("medians.pickle", 'rb') as my_file:
        medians = pickle.load(my_file)
        for feature in features:
            X[pd.isna(X[feature])] = medians[feature]['common']
        return np.array(X[features])
    
@_guard_
def get_predictions(thetas, data):
    y_hat = []
    for theta in thetas:
        mdl = MyLR(theta)
        y_hat.append(mdl.predict_(data))
    y_hat = np.c_[y_hat[0], y_hat[1], y_hat[2], y_hat[3]]
    y_hat = np.argmax(y_hat, axis=1).reshape((-1, 1))
    return y_hat

@_guard_
def main(filename):
    data = read_data(filename)
    if data is None:
        print("File reading error!")
        exit()
    features = data.columns.values[6:]
    X = fill_nan(data, features)
    print(X.shape)
    with open("model.pickle", 'rb') as my_file:
        models = pickle.load(my_file)
    predictions = get_predictions(models, X)

    result = pd.DataFrame(predictions).rename(columns={0:"Hogwarts House"}).applymap(lambda x: houses_index[x])
    print(result)
    result.to_csv("houses.csv", index_label='Index')



if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print("Please, provide the filename in the program arguments")
        exit()
    filename = sys.argv[1]
    main(filename)