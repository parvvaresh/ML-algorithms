import numpy as np
from activation_function import unit_step_func

class Perceptron:
    def __init__(self,
                 learning_rate : int,
                 n_iters : int) -> None:
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weight = None
        self.bias = None
    
    def fit(self,
            X : np.array, 
            y : np.array) -> None:
        
        n_samples , n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        y_bianary = np.array([1 if y > 0 else 0 for y_sample in y])
        
        for _ in range(self.n_iters):
            for index, x in enumerate(X):
                linear_output = np.dot(x, self.weight) + self.bias
                y_pred = self.activation_func(linear_output)
                
                error = y_bianary[index] - y_pred
                update = error * self.learning_rate

                self.weight += update * X
                self.bias += update
        

        def predict(self, 
                    X : np.array) -> np.array:
            y_pred = np.array(
                [self._predict(x) for x in x])
            y_pred = self.activation_func(y_pred)
            return y_pred
        
        def _predict(self,
                     x : np.array) -> np.array:
            
            return np.dot(x, self.weight) + self.bias
