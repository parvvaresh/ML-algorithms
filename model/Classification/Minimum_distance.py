from KNN.metrics import *


class Minimum_distance:
    def __init__(self,
                 metric) -> None:
        self.metric = metric
        self.classes = None
        self.centroid = None
    

    def fit(self, 
            X : np.array,
            y : np.array) -> None:
        
        self.classes = np.unique(y)
        self.centroid = list()

        for index, _class in enumerate(self.classes):
            X_class = X[y == _class]
            center = np.mean(X_class, axis=1)
            self.centroid[index] = center
        self.centroid = np.array(self.centroid)
    

    def predict(self,
                X : np.array) -> np.array:
        
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self,
                 x : np.array) -> np.array:
        distance = [self.metric(center, x) for center in self.centroid]
        index_neareset_center = np.argmin(distance)

        return self.classes[index_neareset_center]