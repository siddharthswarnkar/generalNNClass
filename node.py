import numpy as np
import math
import random 

def sigmoid(theta, x):
	return 1/(1 + math.exp(-np.dot(np.array(theta), np.array(x))))

def step(theta, x):
	if np.dot(np.array(theta), np.array(x)) > 0 :
		return 1
	else :
		return 0

def tanh(theta, x):
	t = np.dot(np.array(theta), np.array(x))
	return (math.exp(t) - math.exp(-t))/(math.exp(t) + math.exp(-t)) 

#def softmax(theta, x):


class DotProductError(Exception):
    print('Vectors dimensions does not match')

class node(object):
	def __init__(self, n):
		self.theta = [randrange(-0.1,0.1) for i in range(n)]
		self.activation_func = 'sigmoid'

	def __init__(self, theta, activation_func):
		self.theta = theta
		self.activation_func = activation_func

	def compute_output(self, x):
		if len(x) != len(theta):
			raise DotProductError

		if activation_func == 'sigmoid':
			return sigmoid(theta, x)
		elif activation_func == 'step':
			return step(theta, x)
		elif activation_func == 'tanh':
			return tanh(theta, x)	

if __name__ == '__main__':
	pass			