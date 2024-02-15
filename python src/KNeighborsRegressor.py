import numpy as np
class KNeighborsRegressor:
    def __init__(self, k):
        self.k = k 
    
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for index , x in  enumerate(X):
          distance =  [self._euclidean_distance(x, x_train) for x_train in self.x_train]
          index_sort = np.argsort(distance)[ : self.k]
          y_pred[index] = np.mean(self.y_train[index_sort])
        return y_pred

    def _euclidean_distance(self, x1, x2):
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance
        
