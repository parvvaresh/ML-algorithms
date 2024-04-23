import numpy as np

from tool import sigmoid

class logstic_regression: #this for bianary classifiction
    def __init__(self,
                 learning_rate : float = 1e-5,
                 iter : int = 1000) -> None:
        self.learning_rate = learning_rate
        self.iter = iter 
        
        self.costs = list()
        self.wieghts = None
        self.bias = None
    
    
    def fit(self,
            X_train : np.array,
            y_train : np.array) -> None:
        
        self.n_sampels, self.n_featues = X_train.shape
        self.wieghts = np.zeros(self.n_featues)
        self.bias = 0
        
        self.costs = []
        for _ in range(self.iter):
            y_pred = np.dot(X_train, self.wieghts) + self.bias
            y_pred_sign = sigmoid(y_pred)
            
            loss = -()
            cost = np.sum(loss) / (2 * self.n_sampels)
            self.costs.append(cost)
            
            dw = None
            db = None
            
            
            self.wieghts -= (dw * self.learning_rate)
            self.bias -= (db * self.learning_rate)
    
    
    def predict(self, 
                X_test : np.array) -> np.array:
        pass
            
        
                