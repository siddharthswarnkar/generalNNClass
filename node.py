import munpy as np

def sigmoid(theta, x):
	return dot(np.array(theta), np.array(x))

class DotProductError(Error):
    print('Vectors dimensions does not match')
   	pass	
   	exit(1)

class node(object):
	def __init__(self, theta, activation_func):
		self.theta = theta
		self.activation_func = activation_func

	def compute_output(self, x):
		if len(x) != len(theta)
		raise DotProductError

		if activation_func == 'sigmoid':
			sigmoid(theta, )
		