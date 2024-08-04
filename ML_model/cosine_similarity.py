import numpy as np

def cosine_similarity(a : np.array, b : np.array) -> np.array:
    dot_product = np.dot(a , b)
    norm_a , norm_b = np.linalg.norm(a) , np.linalg(b)
    cosine_similarity = dot_product / (norm_a * norm_b)
    return cosine_similarity




class cosine_similarity_matrix():
    pass
