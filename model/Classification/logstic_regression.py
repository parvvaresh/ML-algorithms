import numpy as np
from .tool import sigmoid

class logstic_regression:
  def __init__(self,
               max_iters = 10000,
               learning_rate = 10e-4) -> None:
    self.max_iters = max_iters
    self.learning_rate = learning_rate

    self.weights = None
    self.bias = None

  def fit(self,
          X_train : np.array,
          y_train : np.array) -> None:
    
    """
      X_train - > (n, m)
      wights -‌> (1, m)
      y_train -> (n , 1)
      y_predict - > (n , 1)
      loss ->‌ (n , 1)
      dw ->‌(1, m)
    """
    n_sampels , n_features = X_train.shape

    y_train = y_train.reshape((n_sampels, 1))

    self.weights = np.zeros((1 , n_features))
    self.bias = 0

    for _ in range(self.max_iters):
      y_linear = np.dot(X_train, self.weights.T) + self.bias
      y_predict = sigmoid(y_linear)

      loss = (y_predict - y_train)

      dw = np.dot(loss.T , X_train) / n_sampels
      db = np.sum(loss) / n_sampels


      self.weights -= dw * self.learning_rate
      self.bias -= db * self.learning_rate



  def predict(self,
              X_test : np.array) -> np.array:
    y_linear = np.dot(X_test, self.weights.T) + self.bias
    y_predict = sigmoid(y_linear)
    y_predict_cls = [1 if  pred >= 0.5 else 0 for pred in y_predict]
    return np.array(
        y_predict_cls
    )