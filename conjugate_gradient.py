import helper as hlp
import numpy as np

def conjugate_gradient(func, x0, args=(), fprime=None, alpha=0.5, scaling_factor=0.8, numIter=1e5, norm_lim=1e-7, epsilon=1e-10, order=2, disp=True, period=10000):
	if fprime == None:
		fprime = hlp.compute_numerical_grad(func, len(x0),epsilon)

	iters = 0
	func_value = func(x0, *args)
	gPrev = np.array(fprime(x0, *args))
	norm_gradient = hlp.vecnorm(gPrev, order)
	xPrev = np.array(x0)

	pPrev = -gPrev

	while norm_gradient > norm_lim and iters < numIter :
		iters +=1

		
		#if disp and (iters%period == 0 or iters == 1):
		#	print("Iter : %d | Function value : %f" %(iters, func_value))

		alp = alpha
		while func(xPrev + alp*pPrev, *args) > func(xPrev, *args):
			alp *= scaling_factor

		xUpdated = xPrev + alp*pPrev
		gUpdated = np.array(fprime(xUpdated,*args))
		betaUpdated = np.dot(gUpdated,gUpdated)/np.dot(gPrev,gPrev)
		pUpdated = -gUpdated + betaUpdated*pPrev
		
		func_value = func(xUpdated, *args)
		norm_gradient = hlp.vecnorm(gUpdated, order)

		pPrev = pUpdated
		gPrev = gUpdated
		xPrev = xUpdated
	if disp and iters%period != 0:
		print("Iter : %d | Function value : %f" %(iters, func_value))
	return xUpdated

if __name__ == "__main__":
	pass