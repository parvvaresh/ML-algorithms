import numpy as np

class k_means:
  def __init__(self, k = 3, max_iter = 1000):
    self.k = k
    self.max_iter = max_iter
    self.clusters = [[] for _ in range(0, self.k)]
    self.centers = []

  def predict(self, data):
    self.m, self.n = data.shape
    self.data = data

    #select random center for frist time
    index_center_random = np.random.choice(self.m, self.k, replace = False)
    self.centers = [data[index] for index in index_center_random]

    for _ in range(0, self.max_iter):
      self.clusters = self._get_clusters(self.centers)
      old_center = self.centers
      self.centers = self._create_new_center(self.clusters)
      if self._check_center(self.centers, old_center):
        return self._get_labels(self.clusters)


  def _get_labels(self, clusters):
    labels =  np.empty(self.m )
    for label, cluster in enumerate(clusters):
      for index in  cluster:
        labels[index] = label
    return labels

  def _check_center(self, new_center, old_center):
    distance = [
        self._euclidean_distance(new_center[index], old_center[index]) for index in range(0, self.k)
    ]
    return np.sum(distance) == 0

  def _create_new_center(self, clusters):
    center = np.zeros((self.k, self.n))
    for index, cluster in enumerate(clusters):
      cluser_mean = np.mean(self.data[cluster], axis=0)
      center[index] = cluser_mean
    return center

  def _get_clusters(self, centers):
    clusters = [[] for _ in range(0, self.k)]
    for index, point in enumerate(self.data):
      closest_center_index = self._get_closest_center(point, centers)
      clusters[closest_center_index].append(index)
    return clusters

  def _get_closest_center(self, point, centers):
    dis = [self._euclidean_distance(center, point) for center in centers]
    closest_center_index =  dis.index(min(dis))
    return closest_center_index

  def _euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

  def plot(self):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, index in enumerate(self.clusters):
      point = self.data[index].T
      ax.scatter(*point)

    for point in self.centers:
      ax.scatter(*point, marker="x", color="black", linewidth=2)

    plt.show()