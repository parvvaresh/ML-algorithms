import numpy as np
class Perceptron:
  def __init__(self, number_of_input = 0,  lr = 10e-5, threshold = 0):
    self.weights = np.random.rand(number_of_input)
    self.lr = lr
    self.threshold = threshold

  def train(self,x , y, epochs):
    for epoch in range(epochs):
      weighted_sum = np.dot(x , self.weights)
      vectorized_square = np.vectorize(self._activate_func)
      predict = vectorized_square(weighted_sum)
      error = y - predict
      self.weights += self.lr * error * x
      print(f"epoch : {epoch} /// error {error}")
  
  def predict(self, x):
    weighted_sum = np.dot(x , self.weights)
    vectorized_square = np.vectorize(self._activate_func)
    return self.vectorized_square(weighted_sum)
  
  def _activate_func(self, weighted_sum):
    return 1 if weighted_sum > self.threshold else 0
