import numpy as np


class w_knn:
    def __init__(self, k):
        self.k = k
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test):
        y_predict = [self._predict(x) for x in x_test]
        return np.array(y_predict)
    
    def _predict(self, x):
        euclidean_distance = [self._euclidean_distance(x, point) for point in self.x_train]
        index_sort = np.argsort(euclidean_distance)[ : self.k]
        labels_weight = {}
        for index in index_sort:
            if self.y_train[index] in labels_weight:
                labels_weight[self.y_train[index]] += (1 / euclidean_distance[index])
            else:
                labels_weight[self.y_train[index]] = (1 / euclidean_distance[index])
        label, value =  max(labels_weight.items(), key=lambda temp : temp[1])
        return label
                
    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

  
    def accuracy(self, y, y_predict):
      return np.sum(y == y_predict) / len(y)
