#!/data/apps/anaconda/2.7-4.3.1/bin/python

import os
from ClearCloud import ClearCloud
from glob import glob
import numpy as np
import scipy.io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import gdal

t0 = date(2013,6,8)
for dd in range(80):
	t_mid = t0+timedelta(days=dd)
	if os.path.isfile('Cloud_free/refl_vi'+t_mid.strftime('%Y%m%d')+'.mat'):
		continue
	print t_mid
	t_start = t_mid - timedelta(days=7)
	t_end = t_mid + timedelta(days=7)

	ndays = (t_end-t_start).days
	#river area
	river_area = gdal.Open('elevation/buffered_rivers.tif').ReadAsArray()

	#dem_arr = io.imread('elevation/dem_new.tif')
	oyscacld = np.array([]).reshape(0,2139,1499)
	miss = 0
	count = 0
	for i in range(ndays):
		for pre in ['MOD','MYD']:
			t = (t_start+timedelta(days=i)).strftime('%Y%m%d')
			res_files = glob('results/'+pre+'_'+t+'_bindem.tif')
			if not res_files:
				miss += 1
				continue
			elif len(res_files) == 1:
				res_file = res_files[0]
			else:
				res_file = [x for x in res_files if 'new' in x][0]
			mod_arr = gdal.Open(res_file).ReadAsArray()
			mod_arr[mod_arr==1]=2
			mod_arr[mod_arr==-1]=1
			mod_arr[river_area==-99]=-99
			oyscacld = np.vstack([oyscacld, mod_arr[np.newaxis,1272:,1900:3399]])
			count += 1

	#L = scipy.ndimage.zoom(oyscacld,(1,2,2),order=0)
	#high_lakes = np.logical_and(oyscacld==2,dem_arr>=300)
	#oyscacld[high_lakes] = 0
	refl_vi = ClearCloud(oyscacld)
	#refl_vi[high_lakes[ndays,:,:]] = 1
	scipy.io.savemat('Cloud_free/refl_vi'+t_mid.strftime('%Y%m%d')+'.mat',mdict={'refl_vi'+t_mid.strftime('%Y%m%d'):refl_vi})