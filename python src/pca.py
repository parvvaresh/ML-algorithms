import numpy as np

class PCA:
  def __init__(self, n_components = 1):
    self.n_components = n_components
  
  def fit(self, data):
    cov_matrix = self._covar(data)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    eigenvectors = eigenvectors.T
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]

    self.components = eigenvectors[0 : self.n_components]
  
  def transform(self, x):
    mean = np.mean(x, axis=0)
    x = x - mean
    return np.dot(x, self.components.T)

  def _covar(self, data):
    result = np.zeros((data.shape[1], data.shape[1]))
    n = data.shape[0]

    for col_index1 in range(data.shape[1]):
      for col_index2 in range(data.shape[1]):
        m_1 = data[: , col_index1] - np.mean(data[: , col_index1])
        m_2 = data[: , col_index2] - np.mean(data[: , col_index2])
        result[col_index1][col_index2] = (np.dot(m_1, m_2.T) / (n - 1))
    return result

  def _Zscore(self, data):
    result = np.zeros((data.shape[0], data.shape[1]))
    for col_index in range(data.shape[1]):
      result[ : , col_index] = (data[: , col_index] - np.mean(data[: , col_index])) / np.std(data[: , col_index])
    return result