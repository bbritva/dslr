import pandas as pd
import matplotlib.pyplot as plt


houses = {
    "Ravenclaw": "#222f5b",
    "Slytherin": "#1A472A",
    "Gryffindor": "#740001",
    "Hufflepuff": "#ecb939"
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


def histogram(data, features):
    for i, feature in enumerate(features):
        fig, axs = plt.subplots()
        for key in houses:

            axs.hist(data[data["Hogwarts House"] == key]
                     [feature].dropna(), alpha=0.6, color=houses[key], label=key)
        axs.legend()
        axs.set_title(feature)
        axs.grid()
        plt.show()


def main():
    data_train = read_data("dataset_train.csv")
    data_test = read_data("dataset_test.csv")
    if data_test is None or data_train is None:
        print("File reading error!")
        exit()
    features = data_train.columns.values[5:]
    histogram(data_train, features=features)


if __name__ == '__main__':
    main()
