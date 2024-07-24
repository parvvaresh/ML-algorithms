import numpy as np

class DecisionStump:
    pass

class Adaboost:
    def __init__(self,
                 n_clf : int) -> None:
        
        self.n_clf = n_clf
        self.clfs = []

    
    def fit(self,
            X : np.array,
            y : np.array) -> None:
        n_samples, n_features = X.shape

        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        for _ in range(self.n_clf):

            cls = DecisionStump()
            min_error = float("inf")

            for feature in range(n_features):
                X_column = X[: , feature]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    pass