import numpy as np
import matplotlib.pyplot as plt

class KNN:
  def __init__(self, k = 3):
    self.k = k
    
  def fit(self, x, y):
    self.x_train = x
    self.y_train = y
    
  def predict(self, x_test):
    y_pred = [self._predict(test) for test in x_test]
    return np.array(y_pred)

  def _distance(self, x_1, x_2):
    return np.sqrt(np.sum((x_1 - x_2) ** 2))

  def _predict(self, x):
    distance = [self._distance(x, train) for train in self.x_train]
    index = list(np.argsort(distance))[ : self.k]
    labels = [self.y_train[i] for i in index]

    most_label = {}
    for label in labels:
      if label in most_label:
        most_label[label] += 1
      else:
        most_label[label] = 1
    
    most_label = dict(sorted(most_label.items(), key = lambda item : item[1]))
    return list(most_label.keys())[-1]

  def accuracy(self, y, y_pred):
    return np.sum(y == y_pred) / len(y)
        

class NaiveBayes:

  def fit(self, X, y):
    n_samples, n_feature = X.shape
    self.classes = np.unique(y)
    n_classes = len(self.classes)

    self._mean = np.zeros((n_classes, n_feature), dtype = np.float64) 
    self._var = np.zeros((n_classes, n_feature), dtype = np.float64) 
    self._possibility = np.zeros(n_classes, dtype = np.float64) 

    for index, Class in enumerate(self.classes):
      X_class = X[y == Class]
      self._mean[index, : ] = X_class.mean(axis = 0)
      self._var[index, : ] = X_class.var(axis = 0)
      self._possibility[index] =  X_class.shape[0] / float(n_samples)
    print("<<model is fit>>")
  
  def predict(self, x_test):
    y_pred = [(self._predict(test)) for test in x_test]
    return np.array(y_pred)
  
  def _predict(self, test):
    possibilities = []
    for index, Class in enumerate(self.classes):
      y_possibility = self._possibility[index]
      x_possibility= self._pdf(index, test)
      result = (y_possibility * x_possibility)
      possibilities.append(result)
    return self.classes[np.argmax(possibilities)]
    
  def _pdf(self, index, test):
    mean = self._mean[index]
    var = self._var[index]
    numerator = np.exp(- ((test - mean) ** 2) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    fainal = numerator / denominator
    result = 1
    for element in fainal:
      result *= element
    return np.array(result)

  def accuracy(self, y, y_pred):
    return np.sum(y == y_pred) / len(y)





class Kmeans:
    def __init__(self, k  = 3, max_iters = 1000, plot_steps = False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        
        self.clusters = [[] for _ in range(0, self.k)]
        self.centers = []
    def predict(self, data):
        self.data = data
        self.n_samples, self.n_features = self.data.shape
        
        index_random_for_centers = np.random.choice(self.n_samples, self.k, replace = False)
        self.centers = [self.data[index] for index in index_random_for_centers]
        
        for _ in range(0, self.max_iters):
            self.clusters = self._get_clusters(self.centers)
            old_centers = self.centers
            self.centers = self._create_new_center(self.clusters) 
            
            if self._check_New_Old_centers(self.centers, old_centers):
                break
        
        
        return self._get_labels(self.clusters)
    
    
    def _get_clusters(self, centers):
        cluster = [[] for _ in range(0, self.k)]
        for index, samples in enumerate(self.data):
            closest_center_index = self.get_closest_center_index(samples, centers)
            cluster[closest_center_index].append(index)
        return cluster
    
    
    def get_closest_center_index(self, point, centers):
        distance = [self._distance_calculation(point, center) for center in centers]
        closest_center_index = distance.index(min(distance))
        return closest_center_index
    
    def _distance_calculation(self, x1, x2):
        return np.sqrt(np.sum((x1- x2) ** 2))
            
    def _create_new_center(self, clusters):
        centers = np.zeros((self.k, self.n_features))
        for index , cluster in enumerate(clusters):
            cluster_mean = np.mean(self.data[cluster], axis = 0)
            centers[index] = cluster_mean
        return centers
    def _check_New_Old_centers(self, new_centers, old_centers):
        distance = [
            self._distance_calculation(new_centers[index], old_centers[index]) for index in range(0, self.k)
        ]
        return np.sum(distance) == 0
            
    
    def _get_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for index , cluster in enumerate(clusters):
            for samples_index in cluster:
                labels[samples_index] = index
        return labels
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.data[index].T
            ax.scatter(*point)

        for point in self.centers:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()

class cosine_similarity:
  def __init__(self):
    print("just use numpy")
  def fit(self, data):
    self.data = np.array(data)
    cosine_similarity_matrix = []

    for samples_vector in self.data:
      temp = []
      for test_samples_vector in self.data:
        temp.append(self._calcute(list(samples_vector), list(test_samples_vector)))
      cosine_similarity_matrix.append(temp)
    return np.array(cosine_similarity_matrix)
    
  
  def _inner_product(self, vector1, vector2):
    if len(vector1) != len(vector2):
      return False
    else:
      sum = 0
      for index in range(0, len(vector1)): #---> or vector2
        sum += (vector1[index] * vector2[index])
      return sum
  
  def _measure_the_vector(self, vector):
    sum = 0
    for element in vector:
      sum += (element ** 2)
    return sum ** 0.5
  
  def _calcute(self, vector1, vector2):
    return (self._inner_product(vector1, vector2)) / ((self._measure_the_vector(vector1)) * (self._measure_the_vector(vector2)))