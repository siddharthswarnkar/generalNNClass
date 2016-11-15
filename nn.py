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
				temp = [nd.node(bias=True)] + [nd.node(inpt=True) for j in range(list_of_layers[i])]
			elif i != num_layers-1:
				temp = [nd.node(bias=True)] + [nd.node(list_of_layers[i-1]+1, activation_func) for j in range(list_of_layers[i])]
			else:
				temp = [nd.node(list_of_layers[i-1]+1, activation_func) for j in range(list_of_layers[i])]			

			self.nodes.append(temp)	

	def layer_output(self, layer_num, prev_layer_output):
		res = []
		if layer_num != self.num_layers-1:
			for i in range(self.list_of_layers[layer_num]+1):
				res.append(self.nodes[layer_num][i].compute_output(prev_layer_output))
			return res
		for i in range(self.list_of_layers[layer_num]):
			res.append(self.nodes[layer_num][i].compute_output(prev_layer_output))
		return res

	def construct_theta_mat(self,layer_num):
		theta_mat = []
		if layer_num != self.num_layers-1:
			for i in range(1,self.list_of_layers[layer_num]+1):
				theta_mat.append(self.nodes[layer_num][i].get_theta())
			return theta_mat
		for i in range(self.list_of_layers[layer_num]):
			theta_mat.append(self.nodes[layer_num][i].get_theta())
		return theta_mat

	def forward_propogation(self, x):
		mat = [[1]+x]	
		for i in range(1,self.num_layers):
			list_of_input = self.layer_output(i,mat[-1])
			mat.append(list_of_input)
		return mat

	def change_network_theta(self, list_of_theta_mat):
		for i,theta_mat in enumerate(list_of_theta_mat[:-1]):
			for j,node in enumerate(self.nodes[i+1][1:]):
				node.change_theta(theta_mat[j])
		for j,node in enumerate(self.nodes[-1]):
				node.change_theta(list_of_theta_mat[-1][j])

	def roll_mat(self,list_of_mat):
		vector = []
		for mat in list_of_mat:
			for row in mat:
				for col in row:
					vector.append(col)
		return vector

	def unroll_vector(self,theta_vector):
	    list_of_theta_mat = []
	    start = 0
	    list_of_layers = self.list_of_layers
	    for i in range(1,self.num_layers):
	        theta_mat = []
	        size = list_of_layers[i]*(list_of_layers[i-1]+1)
	        theta_rolled = theta_vector[start:start+size]
	        for row in range(list_of_layers[i]):
	            temp = []
	            for col in range(list_of_layers[i-1]+1):
	                temp.append(theta_rolled[row*(list_of_layers[i-1]+1) + col])
	            theta_mat.append(temp)
	        list_of_theta_mat.append(theta_mat)
	        start += size
	    return list_of_theta_mat
	
	def predict(self, x, give_confidence=False):
		prev_layer_output = [self.nodes[0][0].compute_output()]
		for i in range(1,self.list_of_layers[0]+1):
			prev_layer_output.append(self.nodes[0][i].compute_output(x[i-1]))
	
		for i in range(1,self.num_layers):
			prev_layer_output = self.layer_output(i, prev_layer_output)

		if not give_confidence:
			max_pos = 0
			for i in range(self.list_of_layers[-1]):
				if prev_layer_output[max_pos] < prev_layer_output[i]:
					max_pos = i
			res = []
			for i in range(self.list_of_layers[-1]):
				if max_pos != i:
					res.append(0)
				else:
					res.append(1)	
			return res
		
		return prev_layer_output

	def compute_cost(self, data, target, lambd=0.5):
		num_data = len(data)
		num_output_vec = self.list_of_layers[-1]

		def cost(theta):
			if self.activation_func == 'sigmoid':
				activation_func = getattr(nd,'sigmoid')
			elif self.activation_func == 'tanh':
				activation_func = getattr(nd, 'tanh')

			list_of_theta_mat = self.unroll_vector(theta)
			self.change_network_theta(list_of_theta_mat)

			result = 0
			for i in range(num_data):
				output = self.predict(data[i], give_confidence=True)
				for j in range(self.list_of_layers[-1]):
					result += (-1/num_data)*(target[i][j]*math.log(output[j]) + (1-target[i][j])*math.log(1-output[j]) )

			for theta_mat in list_of_theta_mat:
				for elem_list in theta_mat:
					for elem in elem_list:
						result += (lambd/(2*num_data))*elem**2 
			return result
		return cost

	def back_propogation(self, data, target, regularization=0.5):
		"""Data consist of list of list of trainging data
		each row is an example"""
		len_of_input = len(data)
		def grad(theta_vector):
			if self.activation_func == 'sigmoid':
				activation_prime = getattr(nd,'sigmoid_prime')
			elif self.activation_func == 'tanh':
				activation_prime = getattr(nd, 'tanh_prime')

			list_of_theta_mat = self.unroll_vector(theta_vector)
			self.change_network_theta(list_of_theta_mat)
			delta = []
			Delta = []
			for matrix in list_of_theta_mat:
				row,col = np.matrix(matrix).shape
				delta.append(np.zeros((row,col)))
				Delta.append(np.zeros((row,col)))
			Delta = np.array(Delta)
			delta = np.array(delta)

			error_matrix = [0]*(self.num_layers)
			for index,input_array in enumerate(data):
				activation_matrix = self.forward_propogation(input_array)
				error_matrix[self.num_layers-1] = np.array(activation_matrix[-1]) - np.array(target[index])
				error_matrix[self.num_layers-2] = np.dot(np.array(list_of_theta_mat[self.num_layers-2]).T,error_matrix[self.num_layers-2+1])[1:]*activation_prime(list_of_theta_mat[self.num_layers-2],activation_matrix[self.num_layers-2])
				print('sjfs',error_matrix[self.num_layers-2])
				print(activation_prime(list_of_theta_mat[self.num_layers-2],activation_matrix[self.num_layers-2]))
				for i in range(self.num_layers-3,0,-1):
					print('matrix \n',np.array(list_of_theta_mat[i]).T)
					print('\nerror_matrix[i+1]\n', error_matrix[i+1])
					error_matrix[i] = np.dot(np.array(list_of_theta_mat[i]).T,error_matrix[i+1][1:])*activation_prime(list_of_theta_mat[i],activation_matrix[i])
					print('\n error', error_matrix[i]	)
					delta[i] = delta[i] + np.matrix(error_matrix[i+1]).T*np.matrix(activation_matrix[i])
			#Delta = delta+regularization*np.array(list_of_theta_mat)
			for ind, del_mat in enumerate(Delta):
				Delta[ind][1:] = delta[1:] + regularization*np.array(list_of_theta_mat)[ind][1:]
			return roll_mat(Delta)
		return grad
