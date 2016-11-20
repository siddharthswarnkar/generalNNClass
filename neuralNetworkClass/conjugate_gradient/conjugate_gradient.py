import neuralNetworkClass.helper as hlp
import numpy as np


def conjugate_gradient(
        func,
        x0,
        args=(),
        fprime=None,
        alpha=0.5,
        scaling_factor=0.8,
        numIter=1e5,
        norm_lim=1e-7,
        epsilon=1e-10,
        order=2,
        disp=False,
        period=10000):
    """
    Conjugate descent algorithm to optimize the cost function using Fletcher-Reeves Update
    Usage:
            func : function
                function to be optimized
            x0 : list
                initial guess
            args : 
                other arguments to be passed to func
            fprime : function
                derivative or jacobian of func, if not passed then will be generated automatically
            alpha : float
                learning rate
            scaling_factor : float
                factor to multiply alpha with when doing line search
            numIter : int
                number of iterations
            norm_lim : float
                minimum value of norm
            epsilon : float
                delta x for calculting fprime
            order : int
                order of norm, max value = Inf
    Example:
            >>> def func(x):
                    return pow(x[0]-2,6.0)+pow(x[1]-3,6.0)
            >>> x0 = [1,2]
            >>> point_of_optima = conjugate_gradient(func,x0)
    """

    if fprime is None:
        fprime = hlp.compute_numerical_grad(func, len(x0), epsilon)

    iters = 0
    func_value = func(x0, *args)
    gradient_prev = np.array(fprime(x0, *args))
    norm_gradient = hlp.vecnorm(gradient_prev, order)
    xPrev = np.array(x0)

    pPrev = -gradient_prev

    while norm_gradient > norm_lim and iters < numIter:
        iters += 1
        if disp and (iters % period == 0 or iters == 1):
            print("Iter : %d | Function value : %f" % (iters, func_value))

        alp = alpha
        while func(xPrev + alp * pPrev, *args) > func(xPrev, *args):
            alp *= scaling_factor

        xUpdated = xPrev + alp * pPrev
        gradient_updated = np.array(fprime(xUpdated, *args))
        betaUpdated = np.dot(gradient_updated, gradient_updated) / \
            np.dot(gradient_prev, gradient_prev)
        pUpdated = -gradient_updated + betaUpdated * pPrev

        func_value = func(xUpdated, *args)
        norm_gradient = hlp.vecnorm(gradient_updated, order)

        pPrev = pUpdated
        gradient_prev = gradient_updated
        xPrev = xUpdated
    if disp and iters % period != 0:
        print("Iter : %d | Function value : %f" % (iters, func_value))
    return xUpdated

if __name__ == "__main__":
    pass
