# generalNNClass

General Neural Network Class
==============================
This class can be used to train neural network of different configuration of input, hidden layers and output layer.

Different optimization and and activation functions have been provided such as:

- Optimization

  + Conjugate Gradient Descent 
  + Gradient Descent

- Activation Finctions

  + Sigmoid
  + Tanh

Usage

.. code-block:: python

   form nn import *
   neralNetObj = neural_network([2,3,2,4], activation_func='tanh')
   data = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8]]
   target = [[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]]   
   weight = np.array([[[.1,.2,.3],
                     [.4,.5,.6],
                     [.7,.8,.9]],
                   
                   [[.10,.11,.12,.13],
                    [.14,.15,.16,.17]],
                   
                   [[.18,.19,.20],
                    [.21,.22,.23],
                    [.24,.25,.26],
                    [.27,.28,.29]] ])
   neuralNetOjb.change_network_theta(matrix)
   neuralNetObj.train(data,target)
   neuralNetObj.predict([2,3])
