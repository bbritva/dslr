import pandas as pd
import numpy as np
import pickle

def _guard_(func):
    def wrapper(*args, **kwargs):
        try:
            return (func(*args, **kwargs))
        except Exception as e:
            print(e)
            return None
    return wrapper

class DataPreparator():
    @_guard_
    def __init__(self):
        self.houses_index = {
            "Ravenclaw": 0,
            "Slytherin": 1,
            "Gryffindor": 2,
            "Hufflepuff": 3
        }
        self.stats = pickle.load(open("stats.pickle", 'rb'))

    @_guard_
    def prepare_target_values(self, data):
        houses = np.array(data["Hogwarts House"]).reshape((-1, 1))
        Y = np.zeros((houses.shape[0], 5), dtype='int8')
        for i, house in enumerate(houses):
            Y[i][self.houses_index[house[0]]] = 1
        Y[:,4] = np.array([self.houses_index[key] for key in houses.flatten()])
        return Y
    
    @_guard_
    def split_data(self, x):
        np.random.shuffle(x)
        limit_train = int(0.8 * x.shape[0])
        return x[:limit_train,:], x[limit_train:,:]
    
    @_guard_
    def normalize_data(self, X, features):
        for feature in features:
            x_min = self.stats[feature]['min']
            x_max = self.stats[feature]['max']
            X[feature] = (X[feature] - x_min) / (x_max - x_min)
        return X

    @_guard_
    def fill_nan(self, X, features):
        for key in self.houses_index:
            for feature in features:
                X.loc[(X["Hogwarts House"] == key) & pd.isnull(X[feature]), feature] = self.stats[feature][key]
        return X
        

    @_guard_
    def prepare_features(self, data):
        features = data.columns.values[6:]
        X = self.fill_nan(data, features)
        X = self.normalize_data(X, features)
        return np.array(X[features])