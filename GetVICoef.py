#!/data/apps/enthought_python/2.7.3/bin/python

import sys
import numpy as np
import time
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse.linalg as LA

def GetVICoef(c,h,lambda1):
	[nStn, nDim] = c.shape	#nStn = number of stations, namely, known points. nDim = number of dimensions
	start = time.time()
	print 'compute pairwise distance'
	phi = pairwise_distances(c)
	end1 = time.time()
	print 'finished calculating pairwise distance '+str(end1-start)+' s'
	#phi_tril = tril(phi, k=-1)
	#phi_tril.data = np.multiply(np.square(phi_tril.data), np.log(phi_tril.data))
	#phi_tril.setdiag(h[:,0]*lambda1)
	#phi = phi_tril.toarray()+phi_tril.toarray().T
	phi[phi>0] = np.multiply(np.square(phi[phi>0]),np.log(phi[phi>0]))
	if lambda1 != 0:
		phi = phi+np.diag(h[:,0])*lambda1
	CM = np.hstack([np.ones((nStn,1)), c])
	lMatrix = np.vstack([np.hstack([phi, CM]), np.hstack([CM.T, np.zeros((nDim+1, nDim+1))])])
#	lMatrix2=lMatrix.copy()
#	lMatrix2[lMatrix2>0] /= np.max(lMatrix2)
#	lMatrix2[lMatrix2<0] /= np.abs(np.min(lMatrix2))
	y=np.vstack([h, np.zeros((nDim+1,1))])
	print 'number of points '+str(c.shape[0])
	print 'finished creating lMatrix'
	end2 = time.time()
	print str(end2-end1)+' s'
#	coef1 = LA.lsmr(lMatrix2,y,damp=1e-3,atol=1e-8,btol=1e-8,maxiter=200)[0]
	if nStn >= 40000:
		coef = LA.lsmr(lMatrix,y,damp=1e-3,atol=1e-9,btol=1e-9,maxiter=500)[0]
	else:
		if np.linalg.cond(lMatrix) < 10**17:
			coef = np.linalg.solve(lMatrix,y)
		else:
			coef = np.zeros(y.size)
	print 'finished calculating coef'
	end3 = time.time()
	print str(end3-end2)+' s'
	#phi = phi+np.diag(h[:,0])*lambda1
	return coef