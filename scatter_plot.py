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


def scatter_plot(data, features):
    for i in range(len(features) - 1):
        for j in range(i + 1, len(features)):
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            for key in houses:
                curr_data = data[data["Hogwarts House"] == key].dropna()
                plt.scatter(curr_data[features[i]], curr_data[features[j]],
                            marker='o', label="Origin", alpha=0.7, c=houses[key])
            plt.grid()
            plt.legend()
            print(f"{i},{j}: {features[i]} vs {features[j]}")
            plt.show()



def main():
    data_train = read_data("dataset_train.csv")
    data_test = read_data("dataset_test.csv")
    if data_test is None or data_train is None:
        print("File reading error!")
        exit()
    features = data_train.columns.values[6:]
    scatter_plot(data_train, features=features)


if __name__ == '__main__':
    main()
