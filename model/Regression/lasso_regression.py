import numpy as np

class lasso_regression:
    def __init__(self,
                 learning_rate : float = 1e-4,
                 iter : int = 1000,
                 alpha : float = 1e-4,) -> None:
        self.learning_rate = learning_rate
        self.iter = iter
        self.alpha = alpha
        
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
            cost = np.sum(loss ** 2) +  self.alpha * np.sum(np.abs(self.weights)) / (2 * self.n_smaples)
            self.cost.append(cost)
            
            #update the weights and costs
            dw = (np.dot(X_train, loss) + self.alpha * np.sign(self.weights))  / self.n_smaples
            db = None
            
            self.weights -= (self.learning_rate *dw)
            self.bias -= (self.learning_rate * db)
    
    
    def predict(self,
                X_test : np.array) -> None :
        return np.dot(X_test, self.weights) + self.bias
            