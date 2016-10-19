from __future__ import division
from conjugate_gradient import conjugate_gradient as CG
import grad_descent
import hypothesis
import numpy as np
from scipy.optimize import fmin_cg
import unittest
import helper as hlp
import pytest


def simple_func(x):        
	return pow(x[0]-2,6.0)/1000.0+pow(x[1]-3,6.0)/1000.0

def func_grad(x):
	g = [0,0]
	g[0] = 6.0*pow(x[0]-2,5.0)
	g[1] = 6.0*pow(x[1]-3,5.0)
	return np.array(g)

def least_square_cost(x):
	#x[0] = slope, x[1] = intercept
	m = x[0]
	b = x[1]
	data = [(1.2,1.1), (2.3,2.1), (3.0,3.1), (3.8,4.0), (4.7,4.9), (5.9,5.9)]
	summation = 0
	for xi,yi in data:
		summation += (yi-m*xi-b)**2
	return summation/6.0

def some_complex_func(list_of_variable,*args):
	x = list_of_variable[0]
	y = list_of_variable[1]
	return (x-1)**2 + (y-2)**2 + (x**2)*(y**2) + ((x-1)**2)*((y-3)**2) #+ np.exp((x**2)*(y**2))

class TestOptimization(unittest.TestCase):

###################### for conjugate descent #################

	'''def test_something(self):
		x = CG(simple_func,[20,20], alpha=0.5,period=10,norm_lim=1e-7,order=2)
		self.assertAlmostEqual(x[0],2.10, 2)
		self.assertAlmostEqual(x[1],2.90, 2) 

	def test_least_square_cost(self):
		x = CG(least_square_cost,[30,100],alpha=0.5,period=10,norm_lim=1e-7,order=2)
		self.assertAlmostEqual(x[0],1.05,2)
		self.assertAlmostEqual(x[1],-0.14,2)

	def test_some_complex_func(self):
		x = CG(some_complex_func,[30,100],alpha=0.5,period=10,norm_lim=1e-7,order=2)
		self.assertAlmostEqual(x[0], 0.215, 3)
		self.assertAlmostEqual(x[1], 2.31, 2)'''

##################### for gradient descent ##################

	def test_something(self):
		x = grad_descent.grad_descent(simple_func,[20,20], adaptive=True, alpha=0.01,period=10,norm_lim=1e-7,order=2)
		self.assertAlmostEqual(x[0],2.10, 2)
		self.assertAlmostEqual(x[1],2.90, 2) 

	def test_least_square_cost(self):
		x = grad_descent.grad_descent(least_square_cost,[30,100], adaptive=True, alpha=0.05,period=10,norm_lim=1e-7,order=2)
		self.assertAlmostEqual(x[0],1.05,2)
		self.assertAlmostEqual(x[1],-0.14,2)

	def test_some_complex_func(self):
		x = grad_descent.grad_descent(some_complex_func,[30,100], adaptive=True ,alpha=0.05,period=10,norm_lim=1e-7,order=2)
		self.assertAlmostEqual(x[0], 0.215, 3)
		self.assertAlmostEqual(x[1], 2.31, 2)

param = [20,20]
#x2 = fmin_cg(simple_func,param, gtol=1e-7, norm=2)
#print 'fmin_cg', x2
x = grad_descent.grad_descent(simple_func,param, alpha=0.01,period=10,norm_lim=1e-5,order=2)
print 'apdna', x