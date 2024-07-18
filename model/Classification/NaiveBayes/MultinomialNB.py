import numpy as np

class MultinomialNB:
    def __init__(self,
                 alpha : float) -> None:
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob = None
        self.classes_ = None
    

    def fit(self,
            X : np.array,
            y : np.array) -> None:
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)


        class_count = np.zeros(self.n_classes, dtype=np.float64)
        feature_count = np.zeros((self.n_classes, n_features), dtype=np.float64)


        for index, _class in enumerate(self.classes_):
            X_c = X[y == _class]
            class_count[index] = X_c.shape[0]
            feature_count[index, : ] = X_c.sum(axis=0)


        self.class_log_prior_ = np.log(class_count) - np.log(class_count.sum())


        smothed_fc = feature_count + self.alpha
        smothed_cc = smothed_cc.sum(axis=1).reshape((-1, 1))
        self.feature_log_prob_ = np.log(smothed_fc) - np.log(smothed_cc)



    def predict(self,
                X : np.array) -> np.array:
        jill = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jill, axis=1)]
