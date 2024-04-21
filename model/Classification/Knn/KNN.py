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
        labels_Neighbors = [self.y_train[index] for index in index_K_Nearest_Neighbors]
        common_repeat_label = self._get_common_repeat_label(labels_Neighbors)
        return common_repeat_label

    def _get_common_repeat_label(self,
                                 labels_Neighbors : list) -> int:
        counter = {}
        for label in labels_Neighbors:
            if label in counter:
                counter[label] += 1
            else:
                counter[label] = 1
        return max(counter.items(), key=lambda item : item[1])[0]        