import numpy as np
from numpy import Inf
import copy
import math

def compute_numerical_grad(func, num_vars,epsilon=1e-10):
    def grad(x, *args):
    	res = np.array([0.0]*num_vars)
    	x_new = np.array([0.0]*num_vars)
    	for i in range(num_vars):
    		x_new = copy.copy(x)
    		x_new[i] += epsilon  
    		res[i] = (np.array(func(np.array(x_new), *args)) - np.array(func(np.array(x), *args)))/epsilon
        return res
    return grad
'''
def f1(x, m):
    return m[0]*x[0] + m[1]*x[1]**2
def f2(x, m):
    return m[0]*x[0]**2 + m[1]*x[1]**3
grad_f1 = compute_numerical_grad(f1, 2)
grad_f2 = compute_numerical_grad(f2, 2)

print(grad_f1([1,1], [1,1]))
print(grad_f2([1,1], [1,1]))
'''

def vecnorm(x, order=2):
    if order == Inf:
        return np.amax(np.abs(x))
    elif order == -Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x)**order, axis=0)**(1.0 / order)

def grad_descent(func, x0, args=(), fprime=None, alpha=0.02, numIter=1e5, norm=1e-6, epsilon=1e-10, order=2, disp=True):
	if fprime == None :
		fprime = compute_numerical_grad(func, len(x0),epsilon)

	iters = 0
	func_value = func(x0, *args)
	gradient = np.array(fprime(x0, *args))
	norm_gradient = vecnorm(gradient)
	x = np.array(x0)

	while norm_gradient > norm and iters < numIter :
		iters += 1
		if disp :
			print("Iter : %d | Function value : %d" %(iters, func_value	))

		x = x - alpha*gradient
		func_value = func(x, *args)
		gradient = np.array(fprime(x, *args))
		norm_gradient = vecnorm(gradient)
	return x	

def func(x):        
	return pow(x[0]-2,2.0)+pow(abs(x[1])-3,3.0)	

def func_grad(x):
	g = [0,0]
	g[0] = 2.0*x[0]
	g[1] = 3.0*pow(abs(x[1]),2.0)
	if x[1] < 0:
		g[1] = -g[1]
	return g

x = grad_descent(func, [-7,3], order=3)#, fprime=func_grad)
print(x)