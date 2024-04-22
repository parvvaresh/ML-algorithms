import numpy as np

from tool import euclidean_distance

class Minimum_distance:
    def __init__(self):
        self.classes = None
        self.centroid = None
    
    def fit(self, 
            X_train : np.array, 
            y_train : np.array) -> None:
        self.classes = np.unique(y_train)
        self.centroid = list()
        for _class in self.classes:
            class_instance = X_train[y_train == _class]
            center = np.mean(class_instance, axis=0)
            self.centroid.append(center)
            
    
    
    def predict(self, 
                X_test : np.array) -> np.array:
        y_pred = [self._predict(x_test) for x_test in X_test]
        return np.array(
            y_pred)
    
    
    def _predict(self, 
                 x_test : np.array) -> int:
        distance = [euclidean_distance(x_test, center) for center in self.centroid]
        index_nearest_center = np.argmin(distance)
        return self.classes[index_nearest_center]