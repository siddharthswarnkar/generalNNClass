import numpy as np
import math
import helper as hlp

def grad_descent(func, x0, args=(), fprime=None, alpha=0.02, adaptive=False,\
				beta=0.8, numIter=1e5, norm_lim=1e-6, epsilon=1e-10, order=2,\
				disp=True, period=10000):
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
