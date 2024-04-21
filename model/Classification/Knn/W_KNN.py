import numpy as np
from tool import euclidean_distance


class Knn:
    def __init__(self,
                 k : int) -> None :
        self.k  = k
    
    
    def fit(self,
            X_train : np.array,
            y_train : np.array) -> None :
        self.X_train = X_train
        self.y_train = y_train
    
    
    def predict(self,
                X_test : np.array) -> np.array :
        y_pred = [self._predict(x) for x in X_test]
        return np.array(
            y_pred
        )
    
    
    def _predict(self, 
                x_test : np.array) -> int :
        
        distance = [euclidean_distance(x_train, x_test) for x_train in self.X_train]
        index_K_Nearest_Neighbors = np.argsort(distance)[ : self.k]
        
        labels_weight = {}
        for index in index_K_Nearest_Neighbors:
            if self.y_train[index] in labels_weight:
                labels_weight[self.y_train[index]] += (1 / distance[index])
            else:
                labels_weight[self.y_train[index]] = (1 / distance[index])
        
        
        label , weight  = max(labels_weight.items(), key=lambda item : item[1])
        return label

