#!/data/apps/enthought_python/2.7.3/bin/python

import numpy as np
from skimage import measure
from scipy import interpolate, ndimage
import shapely
from shapely.geometry import Polygon
from GetCurveNormals import GetCurveNormals

def GetCP_from_MODIS(modis, ptlimit, np1):
	[nr, nc] = modis.shape
	bw0 = 1*(modis>1)
	bw = np.zeros((nr+2, nc+2))
	bw[1:-1, 1:-1] = bw0
	L = measure.label(bw, connectivity=1)
	N1 = np.max(L)
	cb1 = np.array([]).reshape(2,0)
	cn1 = np.array([]).reshape(2,0)
	#print 'Totally '+str(N1)+' curves will be processed'
	for i in range(1, N1+1):
		imgi = (L==i).astype(np.uint8)
		bi = measure.find_contours(imgi, 0.9)
		bi = np.around(bi[0])
		bi1 = np.array([]).reshape(0,2)
		for ki in range(len(bi)-1):
			if ki == len(bi)-2 :
				if all(bi[ki] == bi[ki+1]):
					bi1 = np.vstack([bi1, bi[ki]])
				else:
					bi1 = np.vstack([bi1, bi[ki]])
					bi1 = np.vstack([bi1, bi[ki+1]])
			else:
				if all(bi[ki] == bi[ki+1]):
					continue
				else:
					bi1 = np.vstack([bi1, bi[ki]])
		#print '**********'+str(i)+':size='+str(len(bi1))
		if (len(bi) < ptlimit):
			continue
		if i >N1:
			ihole = 1
		else:
			ihole = 0
		poly = shapely.geometry.Polygon([bi1[x,:] for x in range(len(bi1))])
		poly_sim = poly.simplify(0.6)
		new_coords = list(poly_sim.exterior.coords)
		points = np.vstack([new_coords]).T
		#print 'size(points): '+str(points.shape[0])+' '+str(points.shape[1])
		if points.shape[1] < 5:
			continue
		if (points.shape[1]>1):
			VN, EN = GetCurveNormals(points, ihole)
			cb1 = np.hstack([cb1, points[:,:-1]])
			cn1 = np.hstack([cn1, points[:,:-1]-0.2*VN])
	cloud = np.zeros((nr+2, nc+2))
	cloud[1:-1,1:-1] = (modis==1).astype(np.uint8)
	distcld = ndimage.distance_transform_edt(1-cloud)
	if cb1.size == 0:
		cb = np.array([]).astype(np.float32)
		cn = np.array([]).astype(np.float32)
	else:
		cb1r = cb1[0,:]
		badri = np.where(np.logical_or(cb1r <=2, cb1r>=nr+1))
		cb1c = cb1[1,:]
		badci = np.where(np.logical_or(cb1c <=2, cb1c>=nc+1))
		zqfun = interpolate.RectBivariateSpline(np.arange(distcld.shape[0]),np.arange(distcld.shape[1]), distcld, s=0.0)
		cb1d = zqfun.ev(cb1c, cb1r)
		ifil = np.where(cb1d<=1.2)
		badi = np.union1d(np.union1d(badri[0], badci[0]), ifil[0])
		cb1 = np.delete(cb1, badi,1)
		cn1 = np.delete(cn1, badi,1)
		cb = cb1.T - 1
		cn = cn1.T - 1
	return np.fliplr(cb), np.fliplr(cn)

