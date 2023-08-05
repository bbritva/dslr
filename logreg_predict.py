import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.metrics import accuracy_score


from my_logistic_regression import MyLogisticRegression as MyLR
from data_preparator import DataPreparator as DP   

houses_index = [
    "Ravenclaw",
    "Slytherin",
    "Gryffindor",
    "Hufflepuff"
]

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
def get_predictions(thetas, data):
    y_hat = []
    for theta in thetas:
        mdl = MyLR(theta)
        y_hat.append(mdl.predict_(data))
    y_hat = np.c_[y_hat[0], y_hat[1], y_hat[2], y_hat[3]]
    y_hat = np.argmax(y_hat, axis=1).reshape((-1, 1))
    return y_hat

@_guard_
def main(filename_test, filename_target):
    data = read_data(filename_test)
    if data is None:
        print("File reading error!")
        exit()
    data_preparator = DP()
    X = data_preparator.prepare_features(data, isTest=True)
    with open("model.pickle", 'rb') as my_file:
        models = pickle.load(my_file)
    predictions = get_predictions(models, X)

    result = pd.DataFrame(predictions).rename(columns={0:"Hogwarts House"}).applymap(lambda x: houses_index[x])
    result.to_csv("houses.csv", index_label='Index')
    print("Predictions saved to 'houses.csv'")
    if filename_target is not None:
        try:
            target = pd.read_csv(filename_target)
            print("\nWrong predictions: ")
            print(result[result["Hogwarts House"] != target["Hogwarts House"]])
            print("\nAmount of wrong predictions:", len(result[result["Hogwarts House"] != target["Hogwarts House"]]))
            print("\nScikit-learn accuracy score:", accuracy_score(target["Hogwarts House"], result["Hogwarts House"]))
        except FileNotFoundError:
            print("Target file reading error!")




if __name__ == '__main__':
    filename_target = None
    filename_test = None
    if len(sys.argv) > 1:
        filename_test = sys.argv[1]
    if len(sys.argv) == 3:
        filename_target = sys.argv[2]
    if filename_test is None:
        print("Please, enter the name of the file to predict\n Usage: python logreg_predict.py <filename>\n or: python logreg_predict.py <filename> <target_filename>")
        exit()
    main(filename_test, filename_target)
