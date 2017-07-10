#!/data/apps/enthought_python/2.7.3/bin/python

import numpy as np

def GetCstPnt(dp, bi, np1):
	n = len(bi)
	ind = []
	for dpi in dp:
		dist = np.square(bi[:,0] - dpi[0])+np.square(bi[:,1] - dpi[1])
		mini = np.argmin(dist)
		ind.append(mini)
	ind = sorted(ind)
	ind2 = []
	for i in range(1,len(ind)):
		yi = np.ceil(np.linspace(ind[i-1],ind[i],np1))
		ind2.append(yi)
	yend = np.fmod(np.ceil(np.linspace(ind[-1],n+ind[0],np1)), n)
	ind2.append(yend)
	ind2 = np.asarray(ind2).astype(np.int16)
	return ind2
	
