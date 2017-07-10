#!/data/apps/enthought_python/2.7.3/bin/python

import numpy as np
from sqdistance import sqdistance
import ctypes
import multiprocessing as mp
from contextlib import closing

def VarInt3d(Pix, c, coef):
	nPix = Pix.shape[0]
	Pixa = mp.Array(ctypes.c_long, nPix*3)
	arr = tonumpyarray(Pixa)
	arr[:] = Pix.flat
	temp = np.empty((nPix,1))
	temp[:] = np.NAN
	Pixval = mp.Array(ctypes.c_long, nPix)
	arr1 = tonumpyarray(Pixval)
	arr1[:] = temp.flat
	with closing(mp.Pool(initializer=init2, initargs=(Pixa, c, coef, Pixval,))) as p2:
		p2.map_async(VariationaInterpolationD, range(nPix))
	p2.join()
	result = tonumpyarray(Pixval).reshape(-1,1)
	return result

def init2(Pixa_, c_, coef_, Pixval_):
	global Pixa, c, coef, Pixval
	Pixa = Pixa_
	c = c_
	coef = coef_
	Pixval = Pixval_

def VariationaInterpolationD(i):
	arr = tonumpyarray(Pixa)
	arr1 = tonumpyarray(Pixval)
	Pix0 = arr[3*i:3*(i+1)]
	phi0 = np.sqrt(sqdistance(c.T, Pix0.reshape(-1,1)))
	phi0[phi0>0] = np.multiply(np.square(phi0[phi0>0]), np.log(phi0[phi0>0]))
	lMatrix_mini = np.hstack([phi0.T, np.ones((1,1)), Pix0.reshape(1,-1)])
	Pixval_scalar = np.dot(lMatrix_mini, coef)
	arr1[i] = Pixval_scalar[0][0]

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())
