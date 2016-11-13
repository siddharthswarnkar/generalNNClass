from conjugate_gradient import conjugate_gradient as cg
from grad_descent import grad_descent as gd
import helper as hlp
import numpy as np
import node as nd
import math

class neural_network(object):
	def __init__(self, list_of_layers, activation_func='sigmoid'):
		num_layers = len(list_of_layers)
		self.num_layers = num_layers
		self.list_of_layers = list_of_layers
		self.activation_func = activation_func
		
		self.nodes = []
		for i in range(num_layers):
			if i == 0:
				temp = [nd.node(inpt=True) for j in range(list_of_layers[i])]
			elif i != num_layers-1:
				temp = [nd.node(list_of_layers[i-1], activation_func) for j in range(list_of_layers[i]-1)]
				temp = [nd.node(bias=True)] + temp
			else:
				temp = [nd.node(list_of_layers[i-1], activation_func) for j in range(list_of_layers[i])]			

			self.nodes.append(temp)	
