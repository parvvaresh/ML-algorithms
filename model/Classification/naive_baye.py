import numpy as np

class naive_bayes:
    def __init__(self) -> None:
        pass

    def fit(self,
            X_train : np.array,
            y_train : np.array) -> None:
        
        self.n_samples , self.n_features = X_train.shape
        self.classes = self.unique(y_train)
        
        #initialize training parameters 
        self.mean_classes = self.zeros((self.num_classes.shape[0] , self.n_features))
        self.variance_classes = self.zeros((self.num_classes[0] , self.n_features))
        self.y_possibilities = self.zeros(self.num_classes)
        
        #set tarain parameters
        for index , _class in enumerate(self.classes):
            class_sample = X_train[y_train == _class]
            self.mean_classes[index, : ] = np.mean(class_sample, axis=0)
            self.variance_classes[index, : ] = np.var(class_sample, axis=0)
            self.class_possibilities[index]  = class_sample.shape[0] / self.n_samples
                
    
    def predict(self,
                X_test : np.array) -> np.array:
        y_pred = [self._predict(x_test) for x_test in X_test]
        return np.array(
                 y_pred)
    
    
    def _predict(self, 
                 x_test : np.array) -> int:
        class_possibilities = []
        for index in range(self.classes.shape[0]):
            class_possibility = self.class_possibilities[index]
            possibility = class_possibility  * self._calculate_possibility(x_test, index)
            class_possibilities.append(possibility)
        return np.argmax(class_possibilities)
    

    def _calculate_possibility(self,
                               x_test : np.array,
                               index : int) -> float:
        mean = self.mean_classes[index]
        var = self.var_classes[index]
        denominator = np.sort(2 * np.pi * var)       
        numerator = np.exp(-((x_test- mean) ** 2) / (2 * var))
        
        result = 1
        for p in (numerator / denominator):
            result *= p
        return result