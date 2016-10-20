from __future__ import division
from conjugate_gradient import conjugate_gradient as CG
import grad_descent
from hypothesis import given, strategies
import numpy as np
from scipy.optimize import fmin_cg
import unittest
import helper as hlp
import pytest


def simple_func(x):        
	return pow(x[0]-2,6.0)+pow(x[1]-3,6.0)

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

	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_simple_func_with_conjugate(self,initial_x, initial_y):
		x = CG(simple_func,[initial_x, initial_y], alpha=0.04,period=10,norm_lim=1e-7,order=2, disp=False)
		self.assertAlmostEqual(x[0],2.0, delta=0.03)
		self.assertAlmostEqual(x[1],3.0, delta=0.03)

	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_least_square_cost_with_conjugate(self,initial_x, initial_y):
		x = CG(least_square_cost,[30,100],alpha=0.5,period=10,norm_lim=1e-7,order=2,disp=False)
		self.assertAlmostEqual(x[0],1.05, delta=0.03)
		self.assertAlmostEqual(x[1],-0.14, delta=0.03)

	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_some_complex_func_with_conjugate(self,initial_x, initial_y):
		x = CG(some_complex_func,[30,100],alpha=0.5,period=10,norm_lim=1e-7,order=2,disp=False)
		self.assertAlmostEqual(x[0], 0.215, delta=0.03)
		self.assertAlmostEqual(x[1], 2.31, delta=0.03)

##################### for gradient descent ##################
	
	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_simple_with_grad_descent(self, initial_x, initial_y):
		x = grad_descent.grad_descent(simple_func,[initial_x, initial_y], adaptive=True, alpha=0.01,period=10,norm_lim=1e-7,order=2)
		self.assertAlmostEqual(x[0],2.0, delta=0.05 )
		self.assertAlmostEqual(x[1],3.0, delta=0.05)

	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_least_square_cost_with_grad_descent(self, initial_x, initial_y):
		x = grad_descent.grad_descent(least_square_cost,[30,100], adaptive=True, alpha=0.05,period=10,norm_lim=1e-7,order=2)
		self.assertAlmostEqual(x[0],1.05, delta=0.03)
		self.assertAlmostEqual(x[1],-0.14, delta=0.03)

	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_some_complex_func_with_grad_descent(self, initial_x, initial_y):
		x = grad_descent.grad_descent(some_complex_func,[30,100], adaptive=True ,alpha=0.05,period=10,norm_lim=1e-7,order=2)
		self.assertAlmostEqual(x[0], 0.215, delta=0.03)
		self.assertAlmostEqual(x[1], 2.31, delta=0.03)

