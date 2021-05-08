import numpy as np
from scipy import stats

class AcquisitionFunction(object):
    """
    Parent class for acquisition functions.
    
    Attributes
    ----------
    func : callable
        A routine to compute the acquisition function. Must have signature
        func(X_new, gpr, **kwargs) -> np.ndarray, with arguments
            X_new : array_like; shape (num_points, input_dimensions)
            gpr : GaussianProcessRegressor
    
    minimize_acquisition : bool
        Specifies whether proposal points are found by minimizing or
        maximizing the acquisition function.
    
    minimize_objective : bool
        Specifies whether the acquisition function is constructed for
        minimization or maximization of an objective function.
    
    params : dict
        Key-value pairs specifying default values for parameters, such as the
        trade-off between exploration and exploitation. The contents of this
        dictionary are handled at the subclass level.
    """
    def __init__(self, func, minimize_acquisition, **kwargs):
        self.func = func
        self.minimize_acquisition = minimize_acquisition
        self.params = kwargs
    
    def __call__(self, X_new, gpr, **kwargs):
        """
        Compute the acquisition function at points X_new.
        
        Arguments
        ---------
        X_new : array_like; shape (num_new_pts, input_dimension)
            Locations at which to compute expected improvement.
            
        gpr : GaussianProcessRegressor
            Regressor object, pre-fit to the sample data via the command
            gpr.fit(X_sample, Y_sample).
        """
        new_kwargs = self.params.copy()
        new_kwargs.update(kwargs)
        return self.func(X_new, gpr, **new_kwargs)
    
    def __repr__(self):
        if self.params:
            s = self.__class__.__name__ + "("
            for key in self.params:
                s += "%s = %r, " % (key, self.params[key])
            return s[:-2] + ")"
        else:
            return self.__class__.__name__ + "()"

EI_defaults = {"delta": 0.01,
               "noisy": False,
               "minimize_objective": True}
class ExpectedImprovement(AcquisitionFunction):
    """
    Expected improvement acquisition function. Amenable to either maximization
    or minimization of an objective function.
    
    Parameters
    ----------
    delta : float
        Trade-off parameter for exploration vs. exploitation. Must be a
        non-negative value. A value of zero corresponds to pure exploitation,
        with more exploration at larger values of delta.
    
    noisy : bool
        If True, assumes a noisy model and predicts the expected outputs at
        the input data, rather than using the observed outputs.
    
    minimize_objective : bool
        Designates whether the objective function is to be minimized or maxi-
        mized. In either case, the expected improvement is defined such that
        its value should be maximized to find a new proposal point.
    """
    def __init__(self, **kwargs):
        for key in EI_defaults:
            if key not in kwargs:
                kwargs[key] = EI_defaults[key]
        
        def EI(X_new, gpr, delta, noisy, minimize_objective):
            """
            Compute the expected improvement at points X_new, from a Gaussian
            process surrogate model fit to observed data (X_sample, Y_sample).
            
            Arguments
            ---------
            X_new : array_like; shape (num_new_pts, input_dimension)
                Locations at which to compute expected improvement.
                
            gpr : GaussianProcessRegressor
                Regressor object, pre-fit to the sample data via the command
                gpr.fit(X_sample, Y_sample).
                
            delta : float
                Trade-off parameter for exploration vs. exploitation. Must be
                a non-negative value. A value of zero corresponds to pure ex-
                ploitation, with more exploration at larger values of delta.
                
            noisy : bool
                If True, assumes a noisy model and predicts the expected
                outputs at X_sample, rather than using Y_sample.
                
            minimize_objective : bool
                Designates whether the objective function is to be minimized
                or maximized. By default, minimization is assumed. In either
                case, the expected improvement is defined such that its value
                should be maximized.
            
            Returns
            -------
            ei : np.ndarray; shape (num_points,)
                The expected improvement at each of the points in X_new.
            """
            if delta < 0.0:
                raise ValueError("Exploration parameter must be non-negative.")
            if minimize_objective:
                best = np.min
                sign = -1.0
            else:
                best = np.max
                sign = 1.0
            
            (mu, sigma) = gpr.predict(X_new, return_std = True)
            if (mu.ndim > 1 and mu.shape[1] > 1) or mu.ndim > 2:
                raise RuntimeError("Invalid shape for predicted "
                                   "mean: %s" % (mu.shape,))
            else:
                mu = mu.flatten()
            sigma = np.maximum(1e-15, sigma.flatten())
            # Bump small variances to prevent divide-by-zero.
            
            if noisy:
                mu_sample = gpr.predict(gpr.X_train_)
                best_y = best(mu_sample)
            else:
                best_y = best(gpr.y_train_)
            
            improvement = sign*(mu - best_y + delta)
            Z = improvement/sigma
            return improvement*stats.norm.cdf(Z) + sigma*stats.norm.pdf(Z)
        
        super().__init__(EI, minimize_acquisition = False, **kwargs)
    
    @property
    def minimize_objective(self):
        return self.params["minimize_objective"]

CB_defaults = {"sigma": 1.96}
class LowerConfidenceBound(AcquisitionFunction):
    """
    Lower confidence bound. This acquisition function may only be used for
    minimizing an objective function.
    
    Parameters
    ----------
    sigma : float
        Trade-off parameter for exploration vs. exploitation. Must be a
        non-negative value. A value of zero corresponds to pure exploitation,
        with more exploration at larger values of sigma.
    """
    def __init__(self, **kwargs):
        for key in CB_defaults:
            if key not in kwargs:
                kwargs[key] = CB_defaults[key]
        
        def LCB(X_new, gpr, sigma):
            """
            Compute the lower confidence bound at points X_new, from a Gaussian
            process surrogate model fit to observed data (X_sample, Y_sample).
            
            Arguments
            ---------
            X_new : array_like; shape (num_new_pts, input_dimension)
                Locations at which to compute confidence bound.
                
            gpr : GaussianProcessRegressor
                Regressor object, pre-fit to the sample data via the command
                gpr.fit(X_sample, Y_sample).
                
            sigma : float
                Trade-off parameter for exploration vs. exploitation. Must be
                a non-negative value. A value of zero corresponds to pure ex-
                ploitation, with more exploration at larger values of sigma.
            
            Returns
            -------
            lcb : np.ndarray; shape (num_points,)
                The lower confidence bound at each of the points in X_new.
            """
            if sigma < 0.0:
                raise ValueError("Exploration parameter must be non-negative.")
            
            (mean, std_dev) = gpr.predict(X_new, return_std = True)
            if (mean.ndim > 1 and mean.shape[1] > 1) or mean.ndim > 2:
                raise RuntimeError("Invalid shape for predicted "
                                   "mean: %s" % (mean.shape,))
            else:
                mean = mean.flatten()
            
            return mean - sigma*std_dev
        
        super().__init__(LCB, minimize_acquisition = True, **kwargs)
    
    @property
    def minimize_objective(self):
        return True

class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper confidence bound. This acquisition function may only be used for
    maximizing an objective function.
    
    Parameters
    ----------
    sigma : float
        Trade-off parameter for exploration vs. exploitation. Must be a
        non-negative value. A value of zero corresponds to pure exploitation,
        with more exploration at larger values of sigma.
    """
    def __init__(self, **kwargs):
        for key in CB_defaults:
            if key not in kwargs:
                kwargs[key] = CB_defaults[key]
        
        def UCB(X_new, gpr, sigma):
            """
            Compute the upper confidence bound at points X_new, from a Gaussian
            process surrogate model fit to observed data (X_sample, Y_sample).
            
            Arguments
            ---------
            X_new : array_like; shape (num_new_pts, input_dimension)
                Locations at which to compute confidence bound.
                
            gpr : GaussianProcessRegressor
                Regressor object, pre-fit to the sample data via the command
                gpr.fit(X_sample, Y_sample).
                
            sigma : float
                Trade-off parameter for exploration vs. exploitation. Must be
                a non-negative value. A value of zero corresponds to pure ex-
                ploitation, with more exploration at larger values of sigma.
            
            Returns
            -------
            ucb : np.ndarray; shape (num_points,)
                The upper confidence bound at each of the points in X_new.
            """
            if sigma < 0.0:
                raise ValueError("Exploration parameter must be non-negative.")
            
            (mean, std_dev) = gpr.predict(X_new, return_std = True)
            if (mean.ndim > 1 and mean.shape[1] > 1) or mean.ndim > 2:
                raise RuntimeError("Invalid shape for predicted "
                                   "mean: %s" % (mean.shape,))
            else:
                mean = mean.flatten()
            
            return mean + sigma*std_dev
        
        super().__init__(UCB, minimize_acquisition = False, **kwargs)
    
    @property
    def minimize_objective(self):
        return False