import neuralNetworkClass.conjugate_gradient as cg
import neuralNetworkClass.grad_descent as gd
import neuralNetworkClass.helper as hlp
import numpy as np
import neuralNetworkClass.node as nd
import math


class neural_network(object):

	def __init__(self, list_of_layers, activation_func='sigmoid'):
		"""
		Inilialize neural network
		
		Input : 
			list_of_layers : list 
				list of number of nodes in each layer starting with input to output layer
			activation_fucn : string
				sigmoid or tanh
		"""

		num_layers = len(list_of_layers)
		self.num_layers = num_layers
		self.list_of_layers = list_of_layers
		self.activation_func = activation_func

		self.nodes = []
		for i in range(num_layers):
			if i == 0:
				temp = [nd.node(bias=True)] + [nd.node(inpt=True)
				                for j in range(list_of_layers[i])]
			elif i != num_layers - 1:
				temp = [nd.node(bias=True)] + [nd.node(list_of_layers[i - 1] + 1,
				                activation_func) for j in range(list_of_layers[i])]
			else:
				temp = [nd.node(list_of_layers[i - 1] + 1, activation_func)
				                for j in range(list_of_layers[i])]

			self.nodes.append(temp)

	def layer_output(self, layer_num, prev_layer_output):
		"""
		
		Input :
			layer_num : 
				layer number of required output
			prev_layer_output : 
				output of the previous layer
		Output:
			Returns the output of the layer_num using activation(theta.T*x)
		"""
		res = []
		if layer_num != self.num_layers - 1:
			for i in range(self.list_of_layers[layer_num] + 1):
				res.append(self.nodes[layer_num][i].compute_output(prev_layer_output))
			return res
		for i in range(self.list_of_layers[layer_num]):
			res.append(self.nodes[layer_num][i].compute_output(prev_layer_output))
		return res

 
	def construct_theta_mat(self,layer_num):
		"""Returns theta matrix of the given layer"""	
		theta_mat = []
		if layer_num != self.num_layers-1:
			for i in range(1,self.list_of_layers[layer_num]+1):
				theta_mat.append(self.nodes[layer_num][i].get_theta())
			return theta_mat
		for i in range(self.list_of_layers[layer_num]):
			theta_mat.append(self.nodes[layer_num][i].get_theta())
		return theta_mat
 

	def forward_propogation(self, x):
		"""
		Performs forward propogation
		
		Input : list
			Data point
		Return : list
			returns probability vector
		"""

		mat = [[1] + x]
		for i in range(1, self.num_layers):
			list_of_input = self.layer_output(i, mat[-1])
			mat.append(list_of_input)
		return mat

	def change_network_theta(self, list_of_theta_mat):
		"""
		Changes weights of all the nodes in the network
		
		Input : list of lists
			List of theta matrix corresponding to network
		"""

		for i, theta_mat in enumerate(list_of_theta_mat[:-1]):
			for j, node in enumerate(self.nodes[i + 1][1:]):
				node.change_theta(theta_mat[j])
		for j, node in enumerate(self.nodes[-1]):
				node.change_theta(list_of_theta_mat[-1][j])

	def roll_mat(self, list_of_mat):
		"""
		Rolls the list of theta matrix into one dimension array(vector)
		
		Input : 
			list_of_mat : list
				List of theta matrix of network
		Return : list of list
			vector of theta
		"""
		
		vector = []
		for mat in list_of_mat:
			for row in mat:
				for col in row:
					vector.append(col)
		return vector

	def unroll_vector(self, theta_vector):
		"""
		Converts theta vector into list of theta matrix corresponding to
		network.
		
		Input : 
			theta_vector : list
				vector consisting of theta of all layers
		Output : 
			List of theta matrix like [theta_mat1, theat_mat2, ..]
		"""
		
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
		"""
		Predicts the value of network given new data
		
		Input : 
			x : list
				Input vector i.e. feature vector
			give_confidence : bool
				gives give confidence on each possible output
		Return : list
			if give_confidence is True then : Probability corresponding to each class
			else : Gives a binary vector
		"""

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
		"""
		Cross Entropy Cost function for the network.
		Can be used with sigmoid or tanh
		
		Input : 
			data : data frame (list of lists) 
				Data and corresponding targets
			target : list of lists
				target of respective data points in data
			lambd : float
				regularization factor
		Return : float 
			Cost function which takes theta vector and gives cost value
		"""

		num_data = len(data)
		num_output_vec = self.list_of_layers[-1]

		if self.activation_func == 'sigmoid':
			def cost(theta):
				list_of_theta_mat = self.unroll_vector(theta)
				self.change_network_theta(list_of_theta_mat)

				result = 0
				for i in range(num_data):
					output = self.predict(data[i], give_confidence=True)
					for j in range(self.list_of_layers[-1]):
						result += (-1/num_data)*(target[i][j]*np.log(output[j]) + (1-target[i][j])*np.log(1-output[j]))

				for theta_mat in list_of_theta_mat:
					for elem_list in theta_mat:
						for elem in elem_list[1:]:
							result += (lambd/(2*num_data))*elem**2 
				return result
			
		elif self.activation_func == 'tanh' :
			def cost(theta):
				list_of_theta_mat = self.unroll_vector(theta)
				self.change_network_theta(list_of_theta_mat)

				result = 0
				for i in range(num_data):
					output = self.predict(data[i], give_confidence=True)
					for j in range(self.list_of_layers[-1]):
						result += (-1/num_data)*( target[i][j]* math.log( (output[j]+1)*0.5 ) + ( 1 - target[i][j]) * math.log(1- (output[j]+1)*0.5) )

				for theta_mat in list_of_theta_mat:
					for elem_list in theta_mat:
						for elem in elem_list[1:]:
							result += (lambd/(2*num_data))*elem**2 
				return result
		return cost

	def back_propogation(self, data, target, lambd=0.5):
		"""
		Finds derivative of cost function
		
		Input : 
			data : data frame (list of lists) 
				data and corresponding targets
			target : list of lists
				target of respective data points in data
			lambda : float
				regularization factor
		Return : list
			Gradient function which returns gradient at given theta vector
		"""

		num_data = len(data)

		def grad_cost(theta):
			if self.activation_func == 'sigmoid':
				activation_prime = getattr(nd,'sigmoid_prime')
			elif self.activation_func == 'tanh':
				activation_prime = getattr(nd, 'tanh_prime')

			list_of_theta_mat = self.unroll_vector(theta)
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
			for i in range(num_data):
				activation_matrix = self.forward_propogation(data[i])

				temp_index = self.num_layers - 1
				error_matrix[temp_index] = np.array(activation_matrix[-1]) - np.array(target[i])
				delta[temp_index-1] = delta[temp_index-1] + np.matrix(error_matrix[temp_index]).T*np.matrix(activation_matrix[temp_index-1])
				
				temp_index = self.num_layers - 2
				error_matrix[temp_index] = np.dot(np.array(list_of_theta_mat[temp_index]).T[1:], error_matrix[temp_index+1]) \
											*activation_prime(list_of_theta_mat[temp_index-1], activation_matrix[temp_index-1])
				delta[temp_index-1] = delta[temp_index-1] + np.matrix(error_matrix[temp_index]).T*np.matrix(activation_matrix[temp_index-1])

				for j in range(self.num_layers-3, 0,-1):
					error_matrix[j] = np.dot(np.array(list_of_theta_mat[j]).T[1:], error_matrix[j+1]) \
										*activation_prime(list_of_theta_mat[j-1], activation_matrix[j-1])
					delta[j-1] = delta[j-1] + np.matrix(error_matrix[j]).T*np.matrix(activation_matrix[j-1])					

			for layer in range(self.num_layers - 1):
				row, col = np.matrix(Delta[layer]).shape
				for ind_x in range(row):
					for ind_y in range(col):
						Delta[layer][ind_x, ind_y] = (1/num_data)*(delta[layer][ind_x, ind_y] + lambd*list_of_theta_mat[layer][ind_x][ind_y])
						if ind_y == 0:
							Delta[layer][ind_x, ind_y] = (1/num_data)*delta[layer][ind_x, ind_y]
					
			return self.roll_mat(Delta)	
		return grad_cost
		
	def train(self, data, target, optim_func='gradient_descent', k=5, lambd=0.5):
		"""
		Trains the network for given data and target
		
		Input : 
			data : data frame (list of lists) 
				data and corresponding targets
			target : list of lists
				target of respective data points in data
			optim_func : string 
				Optimization function to train fucntion with.(gradient_descent, conjugate_gradient)
			k : int
				k-fold cross validation 
			lambd : float 
				regularization factor
		"""

		if optim_func == 'gradient_descent':
			optimize = getattr(gd, 'grad_descent')
		elif optim_func == 'conjugate_gradient':
			optimize = getattr(cg, 'conjugate_gradient')			

		num_data = len(data)	
		accuracy = 0
		max_accuracy = 0
		print('k-fold cross-validation, k =', k, '\n')
		for iter_num in range(k):
			print('Iteration for cross-validation i =', iter_num + 1)
			cv_set = []
			train_set = []
			target_cv = []
			target_train = []
			for i in range(num_data):
				if i%k == iter_num:
					cv_set.append(data[i])
					target_cv.append(target[i])
				else:
					train_set.append(data[i])
					target_train.append(target[i])

			cost = self.compute_cost(train_set, target_train, lambd)
			grad_cost = self.back_propogation(train_set, target_train, lambd)

			theta_0 = self.roll_mat([self.construct_theta_mat(j) for j in range(1,self.num_layers)])

			theta = optimize(cost, x0=theta_0, fprime=grad_cost, norm_lim=0.01, alpha=0.05, disp=True, period=100)

			positive = 0
			num_examples = len(cv_set)
			for j in range(num_examples):
				output = self.predict(cv_set[j])
				if output == target_cv[j]:
					positive += 1
			accuracy = positive/num_examples

			if accuracy > max_accuracy :
				theta_optimal = theta
			print('Accuracy :', accuracy)	
		self.change_network_theta(self.unroll_vector(theta_optimal))
