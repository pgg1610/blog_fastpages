import numpy as np
import scipy.optimize as opt
import warnings
from tqdm import trange

def bayesian_optimization(f, gpr, acq_func, bounds, max_iter = None,
                          prop_kwargs = None, minimize = None,
                          verbose = False, noise=0.0):
    """
    Implement Bayesian optimization to maximize or minimize a scalar function.
    
    Arguments
    ---------
    f : callable
        A routine to compute the objective function. Must have signature
        f(X) -> np.ndarray. If a matrix is provided, the routine should
        evaluate the objective on the rows of the matrix.
    
    gpr : GaussianProcessRegressor
        Regressor object, pre-fit to the sample data via the command
        gpr.fit(X_sample, Y_sample).
    
    acq_func : AcquisitionFunction
        The acquisition function to use for generating proposal points.
        Whether the routine seeks a maximizer or a minimizer is determined by
        the value of acq_func.minimize_objective.
    
    bounds : array_like; shape (2, input_dimension)
        Lower and upper bounds defining a box in input space, on which the
        acquisition function is to be optimized. In particular, bounds[0,:]
        should contain the lower bounds and bounds[:,1] the corresponding
        upper bounds. Initial conditions for the optimization are sampled
        uniformly at random over this box.
    
    max_iter : int, optional
        Maximum number of iterations to perform (new points to sample). If not
        specified, uses gpr.X_train_.shape[0].
        
    prop_kwargs : dict, optional
        Key-value pairs specifying values for optional parameters in the
        propose_location subroutine. The value of return_data is forced False.
    
    minimize : bool, optional
        Explicitly specifies whether the objective function is to be minimized.
        If no value is provided, the task is inferred from the acquisition
        function. If a value is provided that contradicts that implied by the
        acquisition function, an Exception is raised.
    
    verbose : bool, optional
        Whether to display a progress bar counting completed iterations.
    
    Returns
    -------
    results : dict, with the following entries:
        "X": np.ndarray of shape (N,d), where rows represent sampled points;
        "y": np.ndarray of shape (N,1), where entries specify measurements of
             the objective function at the corresponding sample point;
        "args": dictionary containing the values passed to the arguments
                gpr, bounds, and prop_kwargs;
        "warnings": list of warning messages raised during the optimization.
    """
    if prop_kwargs is None:
        prop_kwargs = dict()
        prop_kwargs["return_data"] = False
    
    if minimize is None:
        minimize = acq_func.minimize_objective

    elif int(minimize) + int(acq_func.minimize_objective) == 1:
        raise ValueError("Acquisition function contradicts 'minimize' value.")
    
    if gpr.X_train_.shape[0] == gpr.y_train_.size:
        N0 = gpr.y_train_.size #Initial dimensions of points 
    else:
        raise ValueError("Input and output sample shapes don't match.")
        
    if max_iter is None:
        max_iter = N0

    elif max_iter <= 0:
        raise ValueError("Number of iterations must be positive.")


    print(max_iter)
    X = np.r_[gpr.X_train_, np.zeros((max_iter, gpr.X_train_.shape[1]))]
    y = np.r_[gpr.y_train_.flatten(), np.zeros(max_iter)].reshape(-1, 1)
    
    if verbose:
        pbar = trange(N0, N0 + max_iter, ncols = 75)
    else:
        pbar = range(N0, N0 + max_iter) # OG dim + num of iters 
    
    each_iter = {'0':{"X":X, "y":y}}

    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        for i, j in enumerate(pbar):
            # Fitting the GPR on new points -- refitting will it cause overfitting?
            
            gpr.fit(X[:j,:], y[:j])
            next_X = propose_location(acq_func, gpr, bounds, **prop_kwargs)
            X[j,:] = np.array(next_X["best_X"])
            y[j] = f(next_X["best_X"], noise)
            each_iter['{}'.format(i)] = {"X":X[j,:], "y":y[j]}

    args = {"bounds": bounds, "gpr": gpr, "prop_kwargs": prop_kwargs}

    return {"args": args, "warnings": w, "X": X, "y": y}, each_iter

def propose_location(acq_func, gpr, bounds, generator = None,
                     num_sample = 1000, num_improve = 10, return_data = False):
    """
    Locate a subsequent sampling point for Bayesian optimization by optimizing
    an acquisition function in reference to a Gaussian process regression.
    
    acq_func : AcquisitionFunction
        Function to be maximized or minimized when determining which location
        to propose as the next sample point.
        
    gpr : GaussianProcessRegressor
        Regressor object, pre-fit to the sample data via the command
        gpr.fit(X_sample, Y_sample).
    
    bounds : array_like; shape (2, input_dimension)
        Lower and upper bounds defining a box in input space, on which the
        acquisition function is to be optimized. In particular, bounds[0,:]
        should contain the lower bounds and bounds[:,1] the corresponding
        upper bounds. Initial conditions for the optimization are sampled
        uniformly at random over this box.
    
    generator : numpy.random.Generator or int or None
        Generator used for sampling initial conditions or an integer to be
        used as a seed. If None, a new Generator is created that supplies
        unpredictable values.
    
    num_sample : int
        Number of random initial conditions to generate when searching for
        an optimizer of the acquisition function.
    
    num_improve : int
        Among the num_sample initial conditions generated, the num_improve
        points with the best acquisition function values are used as initial
        guesses for an optimization routine. From these, the best resulting
        point is used as the proposal. If num_improve <= 0, the best point is
        selected without any optimization.
    
    return_full : bool
        If False (default behavior), returns only an ndarray representing the
        proposed point. If True, returns a dict with the following entries:
            "best_X": the proposed point;
            "best_y": the target value at best_X;
            "idx": the index within the sample occupied by best_X, best_y;
            "sampled_X": array of row vectors corresponding to sampled points;
            "sampled_y": array of target values at each of the sampled points;
            "target": the (possibly sign-flipped) acquisition function.
    """
    if num_improve > num_sample:
        raise ValueError("Cannot improve more points than sampled: "
                         "%d > %d" % (num_improve, num_sample))
    generator = np.random.default_rng(generator)

    dim = gpr.X_train_.shape[1]
    if acq_func.minimize_acquisition:
        sign = +1.0
    else:
        sign = -1.0
    
    def target(X):
        """
        Evaluate the acquisition function, scaled for minimization.
        """
        X = np.reshape(X, (-1, dim))
        value = sign * acq_func(X, gpr)
        
        # Convert single value to float for scipy.optimize.minimize:
        if np.atleast_1d(value).size == 1:
            return float(value)
        return value
    

    #Generate initial random guesses for the optimization of acq_func
    X_sample = generator.uniform(bounds[0,:], bounds[1,:],
                                 size = (num_sample, dim))
    
    y_sample = target(X_sample)
    
    if num_improve <= 0:
        idx = np.argmin(y_sample, axis = 0)
        propX = X_sample[idx].reshape(-1, dim)
        if return_data:
            return {"best_X": propX,
                    "best_y": float(y_sample[idx]),
                    "idx": idx,
                    "sampled_X": X_sample,
                    "sampled_y": y_sample,
                    "target": target}
        else:
            return propX
    
    idx = np.argsort(y_sample, axis = 0).flatten() # indices: sort increasing
    best_X = X_sample[idx[0]]
    best_y = float(y_sample[idx[0]])

    # Instead of selecting 1 point with optimal value for acq_func -- here the function
    # is optimized starting with that point as the guess. This is done num_imrpvoe times 

    for i in range(num_improve):
        
        # Optimization of the acquisition function from 
        # various starting position as sampled from X_sample[idx[i]]

        res = opt.minimize(target, X_sample[idx[i]],
                           bounds = bounds.T,
                           method = "L-BFGS-B")

        if res.fun < best_y:
            best_X = res.x
            best_y = float(res.fun)
    
    best_X = best_X.reshape(-1, dim)
    if return_data:
        return {"best_X": best_X,
                "best_y": best_y,
                "idx": idx,
                "sampled_X": X_sample,
                "sampled_y": y_sample,
                "target": target}
    else:
        return {"best_X": best_X}