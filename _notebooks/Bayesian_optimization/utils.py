import numpy as np

def random_unitary_matrix(n, seed = None):
    """
    Generate a random unitary matrix of a specified size.
    
    Arguments
    ---------
    n : int
        Size of the requested matrix.
    
    seed : int, optional
        Seed value to be used for generating the random matrix. If not pro-
        vided, unpredictable values will result.
    
    Returns
    -------
    U : np.ndarray; shape (n, n)
        A unitary matrix.
    """
    generator = np.random.default_rng(seed)
    U = generator.normal(size = (n, n))
    return np.linalg.svd(U)[0]

def pre_rotate_objective(func, rotation, eval_on = "rows", **kwargs):
    """
    Generate a new function that applies another specified function after
    first rotating the input within the appropriate domain.
    
    Arguments
    ---------
    func : callable
        Function to be executed after applying the rotation. Can be vectorized
        such that passing a matrix evaluates the function on each row or on
        each column (see eval_on, below).
    
    rotation : int or array_like
        If an array_like, specifies the rotation matrix to apply. If an int,
        specifies the dimensionality of input vectors in order to randomly
        generate a rotation matrix. Regardless of the value of eval_on, the
        rotation is represented by multiplying the matrix on the right by a
        column vector input (y = U*x).
    
    eval_on : {"cols", "rows}
        If "rows" (default), assumes that vectorized evaluations of func
        evaluate the function on the rows of a two-dimensional matrix with
        shape (num_points, num_dimensions). Otherwise, assumes the function
        is evaluated on columns of a (num_dimensions, num_points) matrix.
    
    Additional keyword arguments are used as parameter values and passed to
    func during the underlying evaluations.
    
    It is assumed that the function evaluated by func uses more than a single
    scalar input. E.g., if eval_on == "cols", then the argument passed to the
    rotated function should have shape (n,) for a single vector or (n, p) for
    a matrix of row vectors, with n > 1 and p > 0.
    
    Returns
    -------
    rotated_func : callable
        A new function that evaluates the original one after applying the
        generated rotaion to the input matrix.
    
    rotation : np.ndarray
        Matrix applied to arguments before evaluating via func. Regardless of
        the value of eval_on, the rotation is represented by multiplying the
        matrix on the right by a column vector input (y = U*x).
    """
    if eval_on not in ["rows", "cols"]:
        raise ValueError("Unrecognized value for eval_on: %s" % (eval_on,))
    
    if isinstance(rotation, int):
        rotation = random_unitary_matrix(rotation)
    else:
        rotation = np.asarray(rotation)
    
    if eval_on == "rows":
        def rotated_func(X):
            X = np.atleast_2d(X)
            return func(np.matmul(X, rotation.T), **kwargs)
    else:
        def rotated_func(X):
            X = np.atleast_2d(X)
            if X.shape[0] == 1: # A single vector is mapped by np.atleast_2d
                X = X.T # to a row-vector; must transpose to a column.
            return func(np.matmul(rotation, X), **kwargs)
    
    return (rotated_func, rotation)