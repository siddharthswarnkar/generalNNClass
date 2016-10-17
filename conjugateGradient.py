import helper as hlp
import numpy as np
import math

def conjugateGradient(func, x0, args=(), fprime=None, alpha=0.02, scaling_factor=0.8, numIter=1e5, norm_lim=1e-6, epsilon=1e-10, order=2, disp=True, period=10000):
	if fprime == None:
		fprime = hlp.compute_numerical_grad(func, len(x0),epsilon)

	iters = 0
	func_value = func(x0, *args)
	gPrev = np.array(fprime(x0, *args))
	norm_gradient = hlp.vecnorm(gPrev, order)
	xPrev = np.array(x0)

	pPrev = -gPrev

	while norm_gradient > norm_lim and iters < numIter :
		iters +=1

		if disp and iters%period == 0 or iters == 1:
			print("Iter : %d | Function value : %f" %(iters, func_value))
			print(alpha)

		alpha = alpha
		while func(xPrev + alpha*pPrev) > fucn(xPrev):
			alpha *= scaling_factor

		xUpdated = xPrev + alpha*pPrev

		gUpdated = np.array(fprime(xUpdated,*args))
		betaUpdated = np.dot(gUpdated,gUpdated)/np.dot(gPrev,gPrev)
		pUpdated = -gUpdated + betaUpdated*pPrev
		
		func_value = func(xUpdated, *args)
		norm_gradient = hlp.vecnorm(gUpdated, order)

		pPrev = pUpdated
		gPrev = gUpdated
		xPrev = xUpdated
	if disp and iters%period != 0:
			print("Iter : %d | Function value : %f" %(iters, func_value))
	return xUpdated

def func(x):        
	return pow(x[0]-2,2.0)+pow(x[1]-3,4.0) 

def func_grad(x):
	g = [0,0]
	g[0] = 2.0*(x[0]-2)
	g[1] = 4.0*pow(x[1]-3,3.0)
	return g

grad_func = hlp.compute_numerical_grad(func, 2)
x = [-6.640022,-8.27999006]

x = grad_descent(func, [-7,9], fprime=func_grad, adaptive = True)
print('\n', x)