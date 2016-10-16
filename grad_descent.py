import numpy as np

def compute_numerical_grad(func, epsilon=1e-10):
    def grad(x, *args):
        x = np.array(x)
        return (np.array(func(x+epsilon, *args)) - np.array(func(x, *args)))/epsilon
    return grad

'''
def f1(x, m):
    return [m[0]*x[0], m[1]*x[1]**2]
def f2(x, m):
    return [m[0]*x[0]**2, m[1]*x[1]**3]
grad_f1 = compute_numerical_grad(f1)
grad_f2 = compute_numerical_grad(f2)

print(grad_f1([1,1], [1,2]))
print(grad_f2([1,1], [1,2]))
'''

