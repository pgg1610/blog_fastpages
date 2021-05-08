import numpy as np

def egg_carton(x):
    """
    Evaluate an example function of a single variable at one or more points.
    
    Parameters
    ----------
    x : array_like
        The desired input values. Function is vectorized.
    """
    x = np.asarray(x)
    return np.sin(4.25*x) + 0.25*(x - 4.8)**2.0

def rosenbrock(X):
    """
    Evaluate the extended Rosenbrock potential at one or more points.
    
    Parameters
    ----------
    X : array_like; shape (num_points, num_dimensions)
        Matrix consisting of row vectors on which to evaluate the function. If
        a one-dimensional vector is provided, it is treated as a single point
        in X.size dimensions and reshaped accordingly.
    
    Returns
    -------
    y : np.ndarray; shape (num_points,)
        Value of the Rosenbrock potential at each of the specified points.
    """
    X = np.atleast_2d(X)
    x = X[:,0]
    y = X[:,1]
    z = X[:,2:]
    
    base = (x - 1.0)**2.0 + 100.0*(y - x**2.0)**2.0
    if z.size > 0:
        sum_sq = np.sum(z**2.0, axis = 1)
        return base + 100.0*sum_sq
    else:
        return base

def muller_brown(X):
    """
    Evaluate the extended Muller-Brown potential at one or more points.
    
    Parameters
    ----------
    X : array_like; shape (num_points, num_dimensions)
        Matrix consisting of row vectors on which to evaluate the function. If
        a one-dimensional vector is provided, it is treated as a single point
        in X.size dimensions and reshaped accordingly.
    
    Returns
    -------
    y : np.ndarray; shape (num_points,)
        Value of the Muller-Brown potential at each of the specified points.
    """
    X = np.atleast_2d(X)
    x = np.reshape(X[:,0], (-1, 1))
    y = np.reshape(X[:,1], (-1, 1))
    z = X[:,2:]
    
    A = np.array([[-200.0, -100.0, -170.0, 15.0]])/100.0
    a = np.array([[  -1.0,   -1.0,   -6.5,  0.7]])
    b = np.array([[   0.0,    0.0,   11.0,  0.6]])
    c = np.array([[ -10.0,  -10.0,   -6.5,  0.7]])
    u = np.array([[   1.0,    0.0,   -0.5, -1.0]])
    v = np.array([[   0.0,    0.5,    0.5,  1.0]])
    
    partials = A*np.exp(a*(x - u)**2.0 + b*(x - u)*(y - v) + c*(y - v)**2.0)
    base = np.sum(partials, axis = 1)
    if z.size > 0:
        sum_sq = np.sum(z**2.0, axis = 1)
        return base + 10.0*sum_sq
    else:
        return base