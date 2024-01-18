import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.visited = set()
        self.labels =  None
        self.cluster_id = 0
    
    def dbscan(self, x):
        self.labels = np.full(x.shape[0] , -1)
        for index in range(x.shape[0]):
            if index not in self.visited:
                self.visited.add(index)
                neighbors = self._region_query(x, index)
                if len(neighbors)  < self.min_samples:
                    self.labels[index] = -1
                else:
                    self._expand_cluster(x, index, neighbors)
                    self.cluster_id += 1   
    
    def _region_query(self, x, center_index):
        _distance = euclidean_distances(x, x[center_index].reshape[1, -1]).flatten()
        return np.where(_distance <= self.eps)[0]

    def _expand_cluster(self, x, center_id, neighbors):
        self.labels[center_id] = self.cluster_id
        i = 0
        
        while i < len(neighbors):
            neighbor = neighbors[i]
            if neighbor not in self.visited:
                self.visited.add(neighbor)
                new_neighbors = self._region_query(x, neighbor)
                if len(new_neighbors) > self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = self.cluster_id
            
            i += 1
