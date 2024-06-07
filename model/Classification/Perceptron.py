import numpy as np

class Perceptron:
	def __init__(self,
		       learning_rate : float,
		       n_iters : int) -> None:


		self.learning_rate = learning_rate
		self.n_iters = n_iters

		self.weights = None
		self. bias = None

		self.activation_func = self._unit_step_func


	def fit(self,
		X_train : np.array,
		y_train : np.array) -> None:


		n_sampels , n_features = X_train.shape

		self.weights = np.zeros(n_features)
		self.bias = 0

		y_ = np.array([1 if y >= 0 else  0 for y in y_train])

		for _ in range(self.n_iters):

			for index , x_i in enumerate(X_train):

				linear_y = np.dot(x_i, self.weights) + self.bias
				y_pred = self.activation_func(linear_y)

				update = self.learning_rate * (y_[index] - y_pred)

				self.weights += update * x_i
				self.bias += update



	def predict(self,
				x_test : np.array) -> np.array:

		linear_output = np.dot(x_test, self.weights) + self.bias
		y_predicted = self.activation_func(linear_output)
		return y_predicted



	def _unit_step_func(self, 
						x : np.array) -> np.array:
		return np.where(x >= 0, 1, 0)





 
