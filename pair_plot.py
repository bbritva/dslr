import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def pair_plot(data, features):
    features.append("Hogwarts House")
    X = data[features]
    sns.pairplot(X, hue="Hogwarts House", palette=houses)
    plt.show()


def main():
    data_train = read_data("dataset_train.csv")
    data_test = read_data("dataset_test.csv")
    if data_test is None or data_train is None:
        print("File reading error!")
        exit()
    features = data_train.columns.values[6:].tolist()
    pair_plot(data_train, features)


if __name__ == '__main__':
    main()
