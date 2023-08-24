import numpy as np

class linear_regression:
  def __init__(self, learning_rate = 1e-6, repeat_num = 100000):
    self.learning_rate = learning_rate
    self.repeat_num = repeat_num
    self.costs = []
  
  def fit(self, x_train, y_train):
    m , n = x_train.shape
    self.weights = np.zeros((n , 1))
    y_train = y_train.reshape(m, 1)
    self.bias = 0
    for _ in range(0, self.repeat_num):
      y_predict = np.dot(x_train, self.weights) + self.bias
      loss = y_predict - y_train
      cost = np.sum(loss ** 2) / (2 * m)

      self.costs.append(cost)
      dw = np.dot(x_train.T, loss) / m
      db = np.sum(loss) / m
      
      self.weights -= (dw * self.learning_rate)
      self.bias -= (db * self.learning_rate)
  
  def predict(self, x_test):
    y_predict = np.dot(x_test, self.weights) + self.bias
    return y_predict
  
  def show_costs(self):
    import matplotlib.pyplot as plt
    plt.plot(self.costs)