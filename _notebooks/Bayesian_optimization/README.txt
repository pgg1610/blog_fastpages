Bayesian Optimization in Python

This package provides simple routines that leverage the capabilities of the
sklearn.gaussian_process.GaussianProcessRegressor object in order to perform
Bayesian optimization of scalar functions. PyBO is not directly dependent on
the sklearn package, in the sense that the command ``import pybo'' will not
attempt to access sklearn in any way. Instead, the function definitions within
PyBO assume that certain objects have the same methods and attributes as the
GaussianProcessRegressor.

Dependencies
------------
	numpy
	scipy
	tqdm
	warnings
