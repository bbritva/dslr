import pandas as pd
import numpy as np
import sys
import pickle

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


@_guard_
@_to_numpy_
@_drop_nan_
def mean(arr):
    return sum(arr) / len(arr)


@_guard_
@_to_numpy_
@_drop_nan_
def percentile(x, p):
    x.sort()
    i = (len(x) - 1) * p / 100
    if i.is_integer():
        return x[int(i)]
    floor = int(i)
    res = x[floor] * (floor + 1 - i) + x[floor + 1] * (i - floor)
    return res


@_guard_
@_to_numpy_
@_drop_nan_
def std(x):
    res = (sum([(i - mean(x)) ** 2 for i in x]) / len(x)) ** 0.5
    return res


def calc_values(data, feature):
    return np.array([
        len(data[feature]),
        mean(data[feature]),
        std(data[feature]),
        min(data[feature]),
        percentile(data[feature], 25),
        percentile(data[feature], 50),
        percentile(data[feature], 75),
        max(data[feature]),
    ])


def main(filename):
    data = read_data(filename)
    if data is None:
        print("File reading error!")
        exit()
    features = data.columns.values[6:]
    rows = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    decription = pd.DataFrame(columns=features, index=rows)
    for feature in features:
        decription[feature] = calc_values(data, feature)
    print(decription)
    houses = data["Hogwarts House"].unique()
    print("\nMedian values for Hogwarts Houses")
    houses_median = pd.DataFrame(columns=features, index=np.r_[houses,["common", "max", "min"]])
    for feature in features:
        for house in houses:
            houses_median[feature][house] = percentile(data[data["Hogwarts House"] == house][feature], 50)
    houses_median.loc["common"] = np.array(decription.loc[['50%']])
    houses_median.loc["max"] = np.array(decription.loc[['Max']])
    houses_median.loc["min"] = np.array(decription.loc[['Min']])
    print(houses_median)
    with open("stats.pickle", 'wb') as my_file:
        pickle.dump(houses_median, my_file)
        print("All results are saved =)")




if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print("Please, provide the filename in the program arguments")
        exit()
    filename = sys.argv[1]
    main(filename)
