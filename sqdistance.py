#!/data/apps/enthought_python/2.7.3/bin/python

import numpy as np

def sqdistance(A,B):
	m = (np.sum(A, axis=1)+np.sum(B, axis=1))/float(A.shape[1]+B.shape[1])
	A = (np.subtract(A.T, m)).T
	B = (np.subtract(B.T, m)).T
	D = (-2)*np.dot(A.T, B)
	D = np.add(D, np.sum(np.square(B), axis=0))
	D = np.add(D, (np.sum(np.square(A), axis=0)[np.newaxis]).T)
	return D