from __future__ import division
from conjugate_gradient import conjugate_gradient as CG
from hypothesis import given, strategies
import helper as hlp
import grad_descent
import numpy as np
import unittest
import pytest


def simple_func(x):        
	return pow(x[0]-2,6.0)+pow(x[1]-3,6.0)

def grad_of_simple_func(x):
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
	return (x-1)**2 + (y-2)**2 + (x**2)*(y**2) + ((x-1)**2)*((y-3)**2)

def grad_of_some_complex_func(list_of_variable):
	x = list_of_variable[0]
	y = list_of_variable[1]
	grad_x = 2*(x-1) + 2*x*(y**2) + 2*(x-1)*((y-3)**2)
	grad_y = 2*(y-2) + 2*y*(x**2) + 2*(y-3)*((x-1)**2)
	return np.array([grad_x, grad_y])

class TestOptimization(unittest.TestCase):

	###################### for conjugate descent #################
	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_simple_func_with_conjugate(self,initial_x, initial_y):
		x = CG(simple_func,[initial_x, initial_y], alpha=0.04,norm_lim=1e-7)
		self.assertAlmostEqual(x[0],2.0, delta=0.03)
		self.assertAlmostEqual(x[1],3.0, delta=0.03)

	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_least_square_cost_with_conjugate(self,initial_x, initial_y):
		x = CG(least_square_cost,[30,100],alpha=0.5,norm_lim=1e-7)
		self.assertAlmostEqual(x[0],1.05, delta=0.03)
		self.assertAlmostEqual(x[1],-0.14, delta=0.03)

	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_some_complex_func_with_conjugate(self,initial_x, initial_y):
		x = CG(some_complex_func,[30,100],alpha=0.5,norm_lim=1e-7)
		self.assertAlmostEqual(x[0], 0.215, delta=0.03)
		self.assertAlmostEqual(x[1], 2.31, delta=0.03)

	##################### for gradient descent ##################
	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_simple_with_grad_descent(self, initial_x, initial_y):
		x = grad_descent.grad_descent(simple_func,[initial_x, initial_y], adaptive=True, alpha=0.01,norm_lim=1e-7)
		self.assertAlmostEqual(x[0],2.0, delta=0.05 )
		self.assertAlmostEqual(x[1],3.0, delta=0.05)

	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_least_square_cost_with_grad_descent(self, initial_x, initial_y):
		x = grad_descent.grad_descent(least_square_cost,[30,100], adaptive=True, alpha=0.05,norm_lim=1e-7)
		self.assertAlmostEqual(x[0],1.05, delta=0.03)
		self.assertAlmostEqual(x[1],-0.14, delta=0.03)

	@given(strategies.floats(min_value =-20, max_value = 20), strategies.floats(min_value = -20, max_value = 20))
	def test_some_complex_func_with_grad_descent(self, initial_x, initial_y):
		x = grad_descent.grad_descent(some_complex_func,[30,100], adaptive=True ,alpha=0.05,norm_lim=1e-7)
		self.assertAlmostEqual(x[0], 0.215, delta=0.03)
		self.assertAlmostEqual(x[1], 2.31, delta=0.03)

	########################### for compute_numerical_grad ########################
	@given(strategies.floats(min_value=-10000,max_value=10000), strategies.floats(min_value=-10000,max_value=10000))
	def test_compute_numerical_grad(self,x,y):
		numerical_grad_of_some_complex_func = hlp.compute_numerical_grad(some_complex_func,2)
		self.assertAlmostEqual(numerical_grad_of_some_complex_func([x,y]).all(), grad_of_some_complex_func([x,y]).all(), delta=0.1)

	########################## for vector norm callculating function vecnorm#######
	def test_vecnorm(self):
		vector = [1,2,3,4,5]
		self.assertAlmostEqual(hlp.vecnorm(vector),np.sqrt(55))
		self.assertAlmostEqual(hlp.vecnorm(vector,order=np.Inf),5)
		self.assertAlmostEqual(hlp.vecnorm(vector,order=-np.Inf),1)