import numpy as np
from Activator_function import sigmoid

class LogsticRregression:
    def __init__(self, iter = 1000, learning_rate = 1e-5) -> None:
        self.learning_rate = learning_rate
        self.iter = iter
        self.weights = None
        self.bias = None

    def fit(self, X : np.array, y : np.array) -> None:
        n_samples , n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iter):
            linear_model = np.dot(X, self.weights) + self.bias

            y_predict = sigmoid(linear_model)
            #pass

