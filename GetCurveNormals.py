#!/data/apps/enthought_python/2.7.3/bin/python

import numpy as np

def GetCurveNormals(points, ihole):
	points1 = points[:,:-1]
	points2 = points[:,1:]
	E = points2 - points1
	EN = np.vstack([-1*E[1,:], E[0,:]])
	EN = np.divide(EN, np.tile(np.sqrt(np.square(E[0,:])+np.square(E[1,:])),(2,1)))
	EN2 = np.hstack([EN[:,-1].reshape(2,1),EN[:,:-1]])
	VN = (EN+EN2)/2
	VN = np.divide(VN, np.tile(np.sqrt(np.square(VN[0,:])+np.square(VN[1,:])),(2,1)))
	if (ihole ==1):
		VN = -1*VN
		EN = -1*EN
	return VN, EN