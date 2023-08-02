import pandas as pd
import numpy as np
import sys

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


def main(filename):
    data = read_data(filename)
    if data is None:
        print("File reading error!")
        exit()
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
    Y = np.c_[Y, houses]
    X = np.array(data[features])
    print(X)
    print(Y)



if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print("Please, provide the filename in the program arguments")
        exit()
    filename = sys.argv[1]
    main(filename)
