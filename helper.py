import numpy as np
from numpy import Inf
import copy

def compute_numerical_grad(func, num_vars,epsilon=1e-10):
    def grad(x, *args):
        res = np.array([0.0]*num_vars)
        x_new = np.array([0.0]*num_vars)
        for i in range(num_vars):
            x_new = copy.copy(x)
            x_new[i] += epsilon  
            res[i] = (np.array(func(np.array(x_new), *args)) - np.array(func(np.array(x), *args)))/epsilon
        return res
    return grad

'''
def f1(x, m):
    return m[0]*x[0] + m[1]*x[1]**2
def f2(x, m):
    return m[0]*x[0]**2 + m[1]*x[1]**3
grad_f1 = compute_numerical_grad(f1, 2)
grad_f2 = compute_numerical_grad(f2, 2)

print(grad_f1([1,1], [1,1]))
print(grad_f2([1,1], [1,1]))
'''

def vecnorm(x, order=2):
    if order == Inf:
        return np.amax(np.abs(x))
    elif order == -Inf:
        return np.amin(np.abs(x))
    else:
        return np.sum(np.abs(x)**order, axis=0)**(1.0 / order)


if __name__ == "__main__":
    pass       
