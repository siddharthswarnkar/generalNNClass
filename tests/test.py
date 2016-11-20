import neuralNetworkClass
from neuralNetworkClass.conjugate_gradient import conjugate_gradient as CG
from hypothesis import given
import neuralNetworkClass.nn as nn
from hypothesis import strategies
import neuralNetworkClass.helper as hlp
import neuralNetworkClass.grad_descent as grad_descent
import numpy as np
import unittest
import pytest
import neuralNetworkClass.node as nd
import math
import csv, random
import os


def simple_func(x):
    return pow(x[0] - 2, 6.0) + pow(x[1] - 3, 6.0)


def grad_of_simple_func(x):
    g = [0, 0]
    g[0] = 6.0 * pow(x[0] - 2, 5.0)
    g[1] = 6.0 * pow(x[1] - 3, 5.0)
    return np.array(g)


def least_square_cost(x):
    #x[0] = slope, x[1] = intercept
    m = x[0]
    b = x[1]
    data = [(1.2, 1.1), (2.3, 2.1), (3.0, 3.1),
            (3.8, 4.0), (4.7, 4.9), (5.9, 5.9)]
    summation = 0
    for xi, yi in data:
        summation += (yi - m * xi - b)**2
    return summation / 6.0


def some_complex_func(list_of_variable):
    x = list_of_variable[0]
    y = list_of_variable[1]
    return (x - 1)**2 + (y - 2)**2 + (x**2) * \
        (y**2) + ((x - 1)**2) * ((y - 3)**2)


def grad_of_some_complex_func(list_of_variable):
    x = list_of_variable[0]
    y = list_of_variable[1]
    grad_x = 2 * (x - 1) + 2 * x * (y**2) + 2 * (x - 1) * ((y - 3)**2)
    grad_y = 2 * (y - 2) + 2 * y * (x**2) + 2 * (y - 3) * ((x - 1)**2)
    return np.array([grad_x, grad_y])

###################### neural net ###################


class TestNeuralNet(unittest.TestCase):

    def test_neural_net(self):
        obj = nn.neural_network([2, 3, 2, 4])
        data = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]]
        target = [
            [
                1, 0, 0, 0], [
                1, 0, 0, 0], [
                0, 1, 0, 0], [
                    0, 1, 0, 0], [
                        0, 0, 1, 0], [
                            0, 0, 1, 0], [
                                0, 0, 0, 1], [
                                    0, 0, 0, 1]]

        matrix = np.array([[[.1, .2, .3],
                            [.4, .5, .6],
                            [.7, .8, .9]],

                           [[.10, .11, .12, .13],
                            [.14, .15, .16, .17]],

                           [[.18, .19, .20],
                            [.21, .22, .23],
                            [.24, .25, .26],
                            [.27, .28, .29]]])

        ans = np.array([0.002, 0.019, 0.025, 0.001, 0.033, 0.039, 0., 0.05,
                        0.057, 0.085, 0.08, 0.089, 0.092, 0.086, 0.083, 0.092,
                        0.095, 0.355, 0.23, 0.243, 0.371, 0.239, 0.253, 0.386,
                        0.25, 0.264, 0.402, 0.261, 0.275])

        fprime = obj.back_propogation(data, target)
        vec = obj.roll_mat(matrix)
        cal = np.array(fprime(vec))
        self.assertAlmostEqual(cal.all(), ans.all(), delta=1)
        self.assertEqual(
            np.array(
                obj.unroll_vector(vec)).all(),
            np.array(matrix).all())
        obj.change_network_theta(matrix)
        cost_func = obj.compute_cost(data, target)
        cost = cost_func(vec)
        self.assertAlmostEqual(cost, 3.5, delta=1)

        obj = nn.neural_network([2, 3, 2, 4], activation_func='tanh')

        ans = np.array([0.007, 0.024, 0.03, 0.001, 0.033, 0.039, 0., 0.05,
                        0.056, 0.142, 0.132, 0.147, 0.149, 0.127, 0.121, 0.135,
                        0.137, 0.102, 0.061, 0.075, 0.152, 0.076, 0.095, 0.2,
                        0.097, 0.121, 0.246, 0.118, 0.148])

        fprime = obj.back_propogation(data, target)
        vec = obj.roll_mat(matrix)
        cal = np.array(fprime(vec))
        self.assertAlmostEqual(cal.all(), ans.all(), delta=1)
        self.assertEqual(
            np.array(
                obj.unroll_vector(vec)).all(),
            np.array(matrix).all())
        obj.change_network_theta(matrix)
        cost_func = obj.compute_cost(data, target)
        cost = cost_func(vec)
        self.assertAlmostEqual(cost, 4.1, delta=1)

        pwd = os.path.split(os.getcwd())[-1]
        if pwd == 'tests':
        	abspath = os.getcwd() + '/'
        else:
        	abspath = os.getcwd() + '/tests/'

        f = open(abspath + 'train.csv', 'r')
        temp = csv.reader(f)
        train_data = []
        flag = 0
        for row in temp:
            if flag == 0:
                flag = 1
                continue
            train_data.append([float(row[0]), float(row[1])])

        f = open(abspath + 'test.csv', 'r')
        temp = csv.reader(f)
        test_data = []
        flag = 0
        for row in temp:
            if flag == 0:
                flag = 1
                continue
            test_data.append([float(row[0]), float(row[1])])

        f = open(abspath + 'train_target.csv', 'r')
        temp = csv.reader(f)
        target_train = []
        flag = 0
        for row in temp:
            if flag == 0:
                flag = 1
                continue
            target_train.append([float(row[0]), float(row[1])])    

        f = open(abspath + 'test_target.csv', 'r')
        temp = csv.reader(f)
        target_test = []
        flag = 0
        for row in temp:
            if flag == 0:
                flag = 1
                continue
            target_test.append([float(row[0]), float(row[1])])    

        train_data = train_data[0:100]
        target_train = target_train[0:100]

        circle = nn.neural_network([2,4,2], activation_func='tanh')
        circle.train(train_data, target_train)
        positive = 0
        num_examples = len(test_data)
        output = []
        for j in range(num_examples):
            output.append(circle.predict(test_data[j]))
            if output[-1] == target_test[j]:
                positive += 1
        accuracy = positive/num_examples
        self.assertAlmostEqual(accuracy, 0.95, delta=0.1)

###################### node.py ######################


class TestActivation(unittest.TestCase):

    # testing sigmoid, sigmoid_prime
    def test_sigmoid(self):
        theta = np.array([1, 2, 3, 4, 5])
        x = np.array([.5, .6, .7, .8, .9])
        self.assertAlmostEqual(
            nd.sigmoid(
                theta,
                x),
            0.9999898700090192,
            places=10)
        self.assertAlmostEqual(nd.sigmoid(-theta, x),
                               1.0129990980873921e-05, places=10)

    def test_sigmoid_prime(self):
        theta = np.array([1, 2, 3, 4, 5])
        x = np.array([.5, .6, .7, .8, .9])
        sgd = nd.sigmoid(theta, x)
        sgd_theoritical = sgd * (1 - sgd)
        self.assertAlmostEqual(
            sgd_theoritical.all(),
            nd.sigmoid_prime(
                theta,
                x).all())

    # testing tanh, tanh_prime
    def test_tanh(self):
        theta = np.array([1, 2, 3, 4, 5])
        x = np.array([.5, .6, .7, .8, .9])
        self.assertAlmostEqual(
            nd.tanh(
                theta,
                x),
            0.9999999997947623,
            places=10)
        self.assertAlmostEqual(nd.tanh(-theta, x), -
                               0.9999999997947623, places=10)

    def test_tanh_prime(self):
        theta = np.array([1, 2, 3, 4, 5])
        x = np.array([.5, .6, .7, .8, .9])
        tanhval = nd.tanh(theta, x)
        tanh_theoritical = tanhval * (1 - tanhval)
        self.assertAlmostEqual(
            tanh_theoritical.all(),
            nd.tanh_prime(
                theta,
                x).all())

    def test_node(self):
        obj1 = nd.node(5)
        theta = np.array(obj1.get_theta())
        self.assertAlmostEqual(theta.all(), np.array(
            [1, 1, 1, 1, 1]).all(), delta=1)
        obj1.change_theta([1, 2, 3, 4, 5])
        theta = np.array(obj1.get_theta())
        self.assertEqual(theta.all(), np.array([1, 2, 3, 4, 5]).all())
        x = np.array([.5, .6, .7, .8, .9])
        self.assertAlmostEqual(obj1.compute_output(x), nd.sigmoid(theta, x))

        obj2 = nd.node(5, activation_func='tanh')
        theta = np.array(obj1.get_theta())
        self.assertAlmostEqual(theta.all(), np.array(
            [1, 1, 1, 1, 1]).all(), delta=1)
        obj1.change_theta([1, 2, 3, 4, 5])
        theta = np.array(obj1.get_theta())
        self.assertEqual(theta.all(), np.array([1, 2, 3, 4, 5]).all())
        x = np.array([.5, .6, .7, .8, .9])
        self.assertAlmostEqual(
            obj1.compute_output(x), nd.tanh(
                theta, x), delta=0.01)

        obj3 = nd.node(inpt=True)
        self.assertEqual(obj3.compute_output(x).all(), x.all())


###################### grad_descent.py and conjugate_gradient.py #########

class TestOptimization(unittest.TestCase):

    ###################### for conjugate descent #################
    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_simple_func_with_conjugate(self, initial_x, initial_y):
        x = CG(simple_func, [initial_x, initial_y],
               alpha=0.5, norm_lim=1e-7, disp=False)
        self.assertAlmostEqual(x[0], 2.0, delta=0.03)
        self.assertAlmostEqual(x[1], 3.0, delta=0.03)

    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_simple_func_with_conjugate(self, initial_x, initial_y):
        x = CG(simple_func,
               [initial_x,
                initial_y],
               fprime=grad_of_simple_func,
               alpha=0.5,
               norm_lim=1e-7,
               disp=False)
        self.assertAlmostEqual(x[0], 2.0, delta=0.03)
        self.assertAlmostEqual(x[1], 3.0, delta=0.03)

    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_least_square_cost_with_conjugate(self, initial_x, initial_y):
        x = CG(least_square_cost, [30, 100],
               alpha=0.5, norm_lim=1e-7, disp=False)
        self.assertAlmostEqual(x[0], 1.05, delta=0.03)
        self.assertAlmostEqual(x[1], -0.14, delta=0.03)

    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_some_complex_func_with_conjugate(self, initial_x, initial_y):
        x = CG(some_complex_func, [30, 100],
               alpha=0.5, norm_lim=1e-7, disp=False)
        self.assertAlmostEqual(x[0], 0.215, delta=0.03)
        self.assertAlmostEqual(x[1], 2.31, delta=0.03)

    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_some_complex_func_with_conjugate(self, initial_x, initial_y):
        x = CG(some_complex_func,
               [30,
                100],
               fprime=grad_of_some_complex_func,
               alpha=0.5,
               norm_lim=1e-7,
               disp=False)
        self.assertAlmostEqual(x[0], 0.215, delta=0.03)
        self.assertAlmostEqual(x[1], 2.31, delta=0.03)

    ##################### for gradient descent ##################
    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_simple_with_grad_descent(self, initial_x, initial_y):
        x = grad_descent.grad_descent(
            simple_func, [
                initial_x, initial_y], adaptive=True, alpha=0.01, norm_lim=1e-7, disp=False)
        self.assertAlmostEqual(x[0], 2.0, delta=0.05)
        self.assertAlmostEqual(x[1], 3.0, delta=0.05)

    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_simple_with_grad_descent(self, initial_x, initial_y):
        x = grad_descent.grad_descent(simple_func,
                                      [initial_x,
                                       initial_y],
                                      fprime=grad_of_simple_func,
                                      adaptive=True,
                                      alpha=0.5,
                                      norm_lim=1e-7,
                                      disp=False)
        self.assertAlmostEqual(x[0], 2.0, delta=0.05)
        self.assertAlmostEqual(x[1], 3.0, delta=0.05)

    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_least_square_cost_with_grad_descent(self, initial_x, initial_y):
        x = grad_descent.grad_descent(
            least_square_cost, [
                initial_x, initial_y], adaptive=True, alpha=0.5, norm_lim=1e-7, disp=False)
        self.assertAlmostEqual(x[0], 1.05, delta=0.03)
        self.assertAlmostEqual(x[1], -0.14, delta=0.03)

    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_some_complex_func_with_grad_descent(self, initial_x, initial_y):
        x = grad_descent.grad_descent(
            some_complex_func, [
                initial_x, initial_y], adaptive=True, alpha=0.5, norm_lim=1e-7, disp=False)
        self.assertAlmostEqual(x[0], 0.215, delta=0.03)
        self.assertAlmostEqual(x[1], 2.31, delta=0.03)

    @given(
        strategies.floats(
            min_value=-20,
            max_value=20),
        strategies.floats(
            min_value=-20,
            max_value=20))
    def test_some_complex_func_with_grad_descent(self, initial_x, initial_y):
        x = grad_descent.grad_descent(some_complex_func,
                                      [initial_x,
                                       initial_y],
                                      fprime=grad_of_some_complex_func,
                                      adaptive=True,
                                      alpha=0.5,
                                      norm_lim=1e-7,
                                      disp=False)
        self.assertAlmostEqual(x[0], 0.215, delta=0.03)
        self.assertAlmostEqual(x[1], 2.31, delta=0.03)


###################### helper.py ######################

class TestHelpers(unittest.TestCase):

    ########################### for compute_numerical_grad ###################
    @given(
        strategies.floats(
            min_value=-10000,
            max_value=10000),
        strategies.floats(
            min_value=-10000,
            max_value=10000))
    def test_compute_numerical_grad(self, x, y):
        numerical_grad_of_some_complex_func = hlp.compute_numerical_grad(
            some_complex_func, 2)
        self.assertAlmostEqual(numerical_grad_of_some_complex_func(
            [x, y]).all(), grad_of_some_complex_func([x, y]).all(), delta=0.1)

    ########################## for vector norm callculating function vecnorm##
    def test_vecnorm(self):
        vector = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(hlp.vecnorm(vector), np.sqrt(55))
        self.assertEqual(hlp.vecnorm(vector, order=np.Inf), 5)
        self.assertEqual(hlp.vecnorm(vector, order=-np.Inf), 1)

if __name__ == '__main__':
    pass
