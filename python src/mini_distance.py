from sklearn.neighbors import DistanceMetric
import numpy as np

class MinimumDistanceClassifier:
    def __init__(self, distance_metric='euclidean'):
        self.distance_metric = distance_metric
        self.classes = None
        self.centroids = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = []
        for target_class in self.classes:
            class_instances = X[y == target_class]
            centroid = np.mean(class_instances, axis=0)
            self.centroids.append(centroid)

    def predict(self, X):
        predictions = []
        distance_metric = DistanceMetric.get_metric(self.distance_metric)
        for instance in X:
            distances = distance_metric.pairwise([instance], self.centroids)
            min_distance_index = np.argmin(distances)
            predicted_class = self.classes[min_distance_index]
            predictions.append(predicted_class)
        return predictions

    def accuracy(self, y, y_test):
      return np.sum(y == y_test) / len(y)

    def conf_matrix(self, y_pred, test_y):

      conf_matrix =  confusion_matrix(y_pred, test_y)

      fig, ax = plt.subplots(figsize=(5,5), dpi=100)
      display = ConfusionMatrixDisplay(conf_matrix)
      ax.set(title='Confusion Matrix for the Diabetes Detection Model')
      display.plot(ax=ax);

    def get_cluster(self):
      return self.centroids
