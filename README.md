============================
General Neural Network Class
============================

This project aims to provide a class to develop and train neural network. This was created for the course AE663 SDES. Read the further documentation to find methods to use this utility.

To use the class, you need *numpy* to be installed in your system.

To install the utility use the following command line statement
```shell
	pip install git+https://github.com/siddharthswarnkar/generalNNClass
```
Usage
```python
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
```
