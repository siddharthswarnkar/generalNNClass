import numpy as np
from numpy import Inf
import copy


def compute_numerical_grad(func, num_vars, epsilon=1e-10):
    """
    Computes numerical gradiet based on forward difference
    
    Input: 
    func : function
        function to find dirivative of func
    num_vars : int 
        number of variables in the func
    epsilon : float
        difference value
    """

    def grad(x, *args):
        res = np.array([0.0] * num_vars)
        x_new = np.array([0.0] * num_vars)
        for i in range(num_vars):
            x_new = copy.copy(x)
            x_new[i] += epsilon
            res[i] = (np.array(func(np.array(x_new), *args)) -
                      np.array(func(np.array(x), *args))) / epsilon
        return res
    return grad


def vecnorm(x, order=2):
    """computes and return vector norm"""
    if order == Inf:
        return np.amax(np.abs(x))
    elif order == -Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x)**order, axis=0)**(1.0 / order)


if __name__ == "__main__":
    pass
