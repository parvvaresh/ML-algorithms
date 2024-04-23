import numpy as np

import numpy as np

class elasticNet_regression:
    def __init__(self,
                 learning_rate : float = 1e-4,
                 iter : int = 1000,
                 alpha1 : float = 1e-4,
                 alpha2 : float = 1e-4) -> None:
        self.learning_rate = learning_rate
        self.iter = iter
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        self.weights = None
        self.bias = None
        self.costs = list()
    
    
    def fit(self, 
            X_train : np.array,
            y_train : np.array,) -> None:
        
        self.n_smaples , self.n_features = X_train.shape
        self.weights = np.zeros(self.n_smaples)
        self.bias = 0
        
        for _ in range(self.iter):
            y_predict = np.dot(X_train, self.weights) + self.bias        
            loss = y_predict - y_train
            l1_regularization = self.alpha1 * np.sum(np.abs(self.weights))
            l2_regularization = self.alpha2 * np.sum(self.weights ** 2)
            cost = (np.sum(loss ** 2)  +  (l1_regularization) + (l2_regularization)) / (2 * self.n_smaples)
            self.cost.append(cost)
            
            #update the weights and costs
            l1_regularization_term = self.alpha1 * np.sign(self.weights)
            l2_regularization_term = self.alpha2 * self.weights
            dw = (np.dot(X_train, loss) + l1_regularization_term + l2_regularization_term)  / self.n_smaples
            db = np.sum(loss) / self.n_samples

            
            self.weights -= (self.learning_rate *dw)
            self.bias -= (self.learning_rate * db)
    
    
    def predict(self,
                X_test : np.array) -> None :
        return np.dot(X_test, self.weights) + self.bias
            