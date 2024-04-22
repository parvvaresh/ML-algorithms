import numpy as np

class linear_regression:
    def __init__(self, 
                learning_rate : float = 1e-4 ,
                iter : int = 1000) -> None:
        self.learning_rate = learning_rate
        self.iter = iter
        
        self.weights = None
        self.bias = None
        self.costs = list()
    
    
    def fit(self,
            X_train : np.array,
            y_train : np.array) -> None:
        
        self.m , self.n = X_train.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        
        for _ in range(self.iter):
            y_pred = np.dot(X_train , self.weights) + self.bias
            loss = y_pred - y_train
            cost = np.sum((loss ** 2)) / self.m
            self.costs.append(cost)
            
            #update the weights and bias
            dw = np.dot(loss, X_train) / self.m
            db = np.sum(loss) / self.m
            
            self.weights -= dw * self.learning_rate
            self.bias -= db * self.learning_rate
    
    
    def predict(self,
                X_test) ->np.array:
        return np.dot(
            X_test, self.weights) + self.b
    
    def show_costs(self):
        import matplotlib.pyplot as plt
        plt.plot(self.costs)