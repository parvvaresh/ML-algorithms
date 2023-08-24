import numpy as np

class Cosine_similarity:
  def __init__(self):
    pass


  def fit(self, vec_1, vec_2):
    vec_1 = self._normal(vec_1)
    vec_2 = self._normal(vec_2)

    k =  self._dot_product(vec_1, vec_2) / (self._norm(vec_1) * self._norm(vec_2))
    return round(k , 8)

  def _dot_product(self, v1, v2):
    dot_product = 0
    for index in range(len(v1)): # or v2
      dot_product += v1[index] * v2[index]
    return dot_product

  def _normal(self, v):
    if len(v.shape) == 1:
      return v
    else:
      return v[0]

  def _norm(self, v):
    return np.sqrt(np.sum(v ** 2))


class Cosine_similarity_matrix(Cosine_similarity):
  def __init__(self):
    pass

  def get_matrix(self, data):
    self.m, self.n = data.shape
    self.cosine_similarity_matrix = np.zeros((self.m , self.m))
    for row in range(0, self.m):
      for col in range(0, self.m):
        self.cosine_similarity_matrix[row][col] = self.fit(data[row], data[col])
    return  self.cosine_similarity_matrix