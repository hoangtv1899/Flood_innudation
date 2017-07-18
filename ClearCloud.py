#!/data/apps/anaconda/2.7-4.3.1/bin/python

import ctypes
import logging
from datetime import date
import time
import multiprocessing as mp
from multiprocessing import Manager
from contextlib import closing
import numpy as np
import scipy.ndimage.morphology as ndimage
from skimage.filters.rank import median
from skimage.morphology import square
from skimage import morphology
from copy import deepcopy
from GetCP_from_MODIS import GetCP_from_MODIS
from sqdistance import sqdistance
from GetVICoef import GetVICoef
from VarInt3d import VarInt3d

def ClearCloud(oyscacld, t_start, t_end):
	try:
		[n, nr, nc] = oyscacld.shape
	except:
		n = 1
		[nr, nc] = oyscacld.shape
	dn_end1 = (t_end-t_start).days+1
	N = n*nr*nc
	temp = np.empty((n, nr, nc))
	temp[:] = np.NAN
	c = mp.Array(ctypes.c_long, N)
	arr1 = tonumpyarray(c)
	arr1[:] = temp.flat
	h = mp.Array(ctypes.c_long, N)
	arr2 = tonumpyarray(h)
	arr2[:] = temp.flat
	sca = mp.Array(ctypes.c_long, N)
	arr = tonumpyarray(sca)
	arr[:] = oyscacld.flat
	DIST = 1
	start1 = time.time()
	with closing(mp.Pool(initializer=init, initargs=(sca, nr, nc, DIST, h, c,))) as p:
        # many processes access different slices of the same array
		p.map_async(GetCP, range(dn_end1+1))
	p.join()
	h1 = tonumpyarray(h)
	h1 = np.delete(h1, np.where(np.isnan(h1)))
	h1 = h1.reshape(-1,1)
	c1 = tonumpyarray(c)
	c1 = np.delete(c1, np.where(np.isnan(c1)))
	c1 = c1.reshape(-1,3)
	end1 = time.time()
	print 'First loop of ClearCloud: '+str(end1-start1)+' s'
	print c1.shape
	if c1.shape[0] > 40000:
		return 'need to divide array'
	if (c1.size == 0) or (c1.shape[0] <= 5):
		print 'c is empty'
		ndays = dn_end1/2 + 1
		return (oyscacld[ndays,:,:] == 2).astype(np.int8)
	else:
		start2 = time.time()
		ndays = dn_end1/2 + 1
		try:
			coef = GetVICoef(c1, h1, 0)
		except:
			ndays = dn_end1/2 + 1
			return (oyscacld[ndays,:,:] == 2).astype(np.int8)
		#np.save('coef'+str(num), coef)
		end2 = time.time()
		print 'Get VI coef: '+str(end2-start2)+' s'
		sca_vi = np.zeros((n,nr,nc))
		for jk in range(n):
			modi = oyscacld[jk,:,:]
			[cldr, cldc] = np.where(modi==1)
			if cldc.size >0:
				Pix = np.hstack([cldc.reshape(-1,1), cldr.reshape(-1,1), jk*DIST*np.ones((len(cldr),1))])
				result = VarInt3d(Pix, c1, coef)
				scai = modi.copy().astype(np.float32)
				scai[cldr.reshape(-1,1), cldc.reshape(-1,1)] = result[:].reshape(-1,1)
				scai = (scai > 0).astype(np.int8)
				sca_vi[jk,:,:] = scai
			else:
				modi[modi==2] = 1
				sca_vi[jk,:,:] = modi.astype(np.int8)
				continue
		end3 = time.time()
		print 'Second loop '+str(end3-end2)+' s'
	return sca_vi.astype(np.uint8)

def init(sca_, nr1, nc1, DIST1, h_, c_):
	global sca, nr, nc, DIST, h, c
	sca = sca_ # must be inhereted, not passed as an argument
	nr = nr1
	nc = nc1
	DIST = DIST1
	h = h_
	c = c_

def init1(sca_, nr1, nc1, DIST1, coef_, c1_, sca_vi_):
	global sca, nr, nc, DIST, coef, c1, sca_vi
	sca = sca_ # must be inhereted, not passed as an argument
	nr = nr1
	nc = nc1
	DIST = DIST1
	coef = coef_
	c1 = c1_
	sca_vi = sca_vi_

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

def VariationaInterpolation(i):
	arr0 = tonumpyarray(sca_vi)
	arr = tonumpyarray(sca)
	modi = arr[nr*nc*i:nr*nc*(i+1)].reshape(nr, nc)
	[cldr, cldc] = np.where(modi==1)
	Pix = np.hstack([cldc.reshape(-1,1), cldr.reshape(-1,1), i*DIST*np.ones((len(cldr),1))])
	result = VarInt3d(Pix, c1, coef)
	scai = modi.copy().astype(np.float32)
	scai[cldr.reshape(-1,1), cldc.reshape(-1,1)] = result[:].reshape(-1,1)
	scai = (scai > 0).astype(np.uint8)
	arr0[nr*nc*i:nr*nc*(i+1)] =scai.flat

def GetCP(i):
	"""no synchronization."""
	arr1 = tonumpyarray(h)
	arr2 = tonumpyarray(c)
	arr = tonumpyarray(sca)
	modi = arr[nr*nc*i:nr*nc*(i+1)].reshape(nr, nc)
	mods = 1*morphology.remove_small_objects(modi==2, min_size=20, connectivity=1)
	mods = 1*(morphology.remove_small_objects(mods==0, min_size=20, connectivity=1)==0)
	modstemp = np.zeros((nr+2, nc+2))
	modstemp[1:-1,1:-1] = mods
	if np.sum(mods) < 10:
		print 'skipped'
		return
	modstemp = median(modstemp.astype(np.uint8), square(3))
	mods = modstemp[1:-1,1:-1]
	modfi = mods*2 + np.logical_and(modi==1, mods==0).astype(np.uint8)
	[cbi, cni] = GetCP_from_MODIS(modfi, 18, 3)
	if cbi.size == 0:
		return
	[rowb, colb] = np.argwhere(np.isnan(cbi)).T
	[rown, coln] = np.argwhere(np.isnan(cni)).T
	cbi = np.delete(cbi, np.hstack([rowb, rown]),0)
	cni = np.delete(cni, np.hstack([rowb, rown]),0)
	d = sqdistance(cbi.T, cbi.T)
	d = d + np.tril(np.ones(d.shape))
	[dr, dc] = np.where(d<0.9)
	cbi = np.delete(cbi, dc, 0)
	cni = np.delete(cni, dc, 0)
	d = sqdistance(cni.T, cni.T)
	d = d + np.tril(np.ones(d.shape))
	[dr, dc] = np.where(d<0.9)
	cbi = np.delete(cbi, dc, 0)
	cni = np.delete(cni, dc, 0)
	d = sqdistance(cbi.T, cni.T)
	d = d + np.tril(np.ones(d.shape))
	[dr, dc] = np.where(d<0.9)
	cbi = np.delete(cbi, np.hstack([dr, dc]), 0)
	cni = np.delete(cni, np.hstack([dr, dc]), 0)
	if cbi.size ==0:
		print 'no constraint points selected'
		return
	ci = np.vstack([np.hstack([cni, DIST*i*np.ones((cni.shape[0],1))]), np.hstack([cbi, DIST*i*np.ones((cbi.shape[0],1))])])
	hi = np.vstack([np.ones((cni.shape[0],1))*0.1, np.zeros((cbi.shape[0],1))])
	arr1[nr*nc*i:nr*nc*i+len(hi)] = hi.flat
	arr2[nr*nc*i:nr*nc*i+len(hi)*3] = ci.flat

