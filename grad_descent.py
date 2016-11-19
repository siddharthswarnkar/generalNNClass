import numpy as np
import math
import helper as hlp

def grad_descent(func, x0, args=(), fprime=None, alpha=0.02, adaptive=False,\
				beta=0.8, numIter=1e5, norm_lim=1e-6, epsilon=1e-10, order=2,\
				disp=False, period=10000):
	'''
	Gradient descent algorithm to optimize the cost function :
		func = function to be optimized
		x0   = initial guess
		args = other arguments to be passed to func
		fprime = derivative or jacobian of func, if not passed then will be generated automatically
		alpha  = learning rate
		adaptive = boolean, allows variable learning rate
		beta = used only if adaptive is True, back tracking line search
		numIter = number of iterations
		norm_lim = minimum value of norm
		epsilon = delta x for calculting fprime
		order = order of norm, max value = Inf
		disp = boolean, displays the iteration number and function value
		period = period of printing (iteration)
	Example:
		def func(x):        
			return pow(x[0]-2,6.0)+pow(x[1]-3,6.0)
		x0 = [1,2]
		point_of_optima = grad_descent(func,x0)
	'''

	if fprime == None :
		fprime = hlp.compute_numerical_grad(func, len(x0),epsilon)

	iters = 0
	func_value = func(x0, *args)
	gradient = np.array(fprime(x0, *args))
	norm_gradient = hlp.vecnorm(gradient, order)
	x = np.array(x0)

	while norm_gradient > norm_lim and iters < numIter :
		iters += 1
		if disp and (iters%period == 0 or iters == 1):
			print("Iteration : %d | Function value : %f" %(iters, func_value))
		if adaptive:
			alpha = 0.5
			while func(x-alpha*gradient) > func(x) :
				alpha = beta*alpha

		x = x - alpha*gradient
		func_value = func(x, *args)
		gradient = np.array(fprime(x, *args))
		norm_gradient = hlp.vecnorm(gradient, order)
	if disp and iters%period != 0:
		print("Iteration : %d | Function value : %f" %(iters, func_value))
	return x	

if __name__ == '__main__':
	pass
