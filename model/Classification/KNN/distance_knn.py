import numpy as np
from collections import Counter

from metrics import *

class DistanceKnn:
    def __init__(self, 
                 K : int,
                 metric) -> None:
        self.K = K
        self.metric = metric
    

    def fit(self,
            X_train : np.array,
            y_train : np.array) -> None:
        self.X_train = X_train
        self.y_train = y_train
    
    def train(self,
              X_test : np.array) -> np.array:
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)
    
    def _predict(self,
                 x : np.array) -> np.array:
        distance = [self.metric(x , x_train) for x_train in self.X_train]
        index_K_Nearest_Neighbors = np.argsort(distance)[ : self.K]


        lables = dict()

        for index in index_K_Nearest_Neighbors:
            if self.y_train[index] in lables:
                lables[self.y_train[index]] += (1 / distance[index])
            else:
                lables[self.y_train[index]] += (1 / distance[index])


        lable , weight = max(lables.items() , key=lambda item : item[1])
        return lable