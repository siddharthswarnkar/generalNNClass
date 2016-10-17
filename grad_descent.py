import numpy as np
import math

import helper as hlp

def grad_descent(func, x0, args=(), fprime=None, alpha=0.02, adaptive=False, beta=0.8, numIter=1e5, norm_lim=1e-6, epsilon=1e-10, order=2, disp=True, period=10000):
	if fprime == None :
		fprime = hlp.compute_numerical_grad(func, len(x0),epsilon)

	iters = 0
	func_value = func(x0, *args)
	gradient = np.array(fprime(x0, *args))
	norm_gradient = hlp.vecnorm(gradient, order)
	x = np.array(x0)

	while norm_gradient > norm_lim and iters < numIter :
		iters += 1
		if disp and iters%period == 0 or iters == 1:
			print("Iter : %d | Function value : %f" %(iters, func_value))
			print(alpha)

		if adaptive:
			alpha = 0.5
			while func(x-alpha*gradient) > func(x) :
				alpha = beta*alpha

		x = x - alpha*gradient
		func_value = func(x, *args)
		gradient = np.array(fprime(x, *args))
		norm_gradient = hlp.vecnorm(gradient, order)
	if disp and iters%period != 0:
		print("Iter : %d | Function value : %f" %(iters, func_value))
	return x	

def func(x):        
	return pow(x[0]-2,2.0)+pow(x[1]-3,4.0)	

def func_grad(x):
	g = [0,0]
	g[0] = 2.0*(x[0]-2)
	g[1] = 4.0*pow(x[1]-3,3.0)
	return g

grad_func = hlp.compute_numerical_grad(func, 2)
x = [-6.640022,-8.27999006]
'''
print(type(func([x[0]+1e-10, x[1]])))
print(func([x[0]+1e-10, x[1]]), func(x))
print((func([x[0]+1e-10, x[1]]) -func(x) )/1e-10, '\n')
print(grad_func(x))
print(func_grad(x), '\n')
'''
x = grad_descent(func, [-7,9], fprime=func_grad, adaptive = True)
print('\n', x)