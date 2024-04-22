import numpy as np
from tool import euclidean_distance

class knn_regression:
    def __init__(self,
                 k : int) -> None:
        self.k = k
    
    def fit(self, 
            X_train : np.array,
            y__train : np.array) -> None:
        self.X_train = X_train
        self.y_train = y__train
    
    
    def predict(self, 
                X_test : np.array) -> np.array:
        y_pred = [self._predict(x) for x in X_test] 
        return np.array(y_pred)

    
    def _predict(self,
                x : np.array) -> float:
        
        distance = [euclidean_distance(x, x_train) for x_train in self.X_train]
        index_K_Nearest_Neighbors = np.argsort(distance)[ : self.k]
        pred = np.mean([( self.y_train[index] * (1 / distance[index]) )  for index in index_K_Nearest_Neighbors])
        return pred