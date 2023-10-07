def nmf(X, num_components, max_iter=100, tol=1e-4):
    """
    Non-negative Matrix Factorization (NMF) using multiplicative updates.

    Parameters:
    - X: Input non-negative matrix (m x n)
    - num_components: Number of components to factorize X into
    - max_iter: Maximum number of iterations
    - tol: Tolerance to check for convergence

    Returns:
    - W: Basis matrix (m x num_components)
    - H: Coefficient matrix (num_components x n)
    """
    m, n = X.shape
    W = np.random.rand(m, num_components)
    H = np.random.rand(num_components, n)

    for iter in range(max_iter):
        W = W * ((X.dot(H.T)) / (W.dot(H).dot(H.T)))
        H = H * ((W.T.dot(X)) / (W.T.dot(W).dot(H)))
        residual = np.linalg.norm(X - W.dot(H), 'fro')
        if residual < tol:
            break

    return W, H
