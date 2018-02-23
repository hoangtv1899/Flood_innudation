#!/data/apps/enthought_python/7.3.2/bin/python

import os
from glob import glob
import numpy as np
import scipy.io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import gdal

def createTif(fname, res_arr, geom, dtype=gdal.GDT_Int16, ndata=-99):
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(
			fname,
			res_arr.shape[1],
			res_arr.shape[0],
			1,
			dtype,
			options = ['COMPRESS=LZW'])
	dataset.SetGeoTransform(geom)
	outband = dataset.GetRasterBand(1)
	outband.WriteArray(res_arr)
	outband.FlushCache()
	outband.SetNoDataValue(ndata)
	dataset.FlushCache()

def ResampleImage(img,img1,resolution,bounding_box,srcnodata=-99):
	if srcnodata:
		os.system('gdalwarp -overwrite -r cubicspline -srcnodata '+str(srcnodata)+' -dstnodata -99 -tr '+\
					str(resolution)+' '+str(resolution)+\
					' -te '+' '.join([str(x) for x in bounding_box])+' '+img+' '+img1)
	else:
		os.system('gdalwarp -overwrite -r cubicspline -tr '+\
					str(resolution)+' '+str(resolution)+\
					' -te '+' '.join([str(x) for x in bounding_box])+' '+img+' '+img1)

t0 = date(2013,6,8)
#read hand raster
ds_hand = gdal.Open('hand/mississippi_hand_resampled.tif')
hand = ds_hand.ReadAsArray()[1272:,1900:3399]
geom_main = ds_hand.GetGeoTransform()
geom_main = (geom_main[0]+1900*geom_main[1],
				0.00219,
				0.0,
				geom_main[3]+1272*geom_main[5],
				0.0,
				-0.00219)
bb = [geom_main[0],
		geom_main[3]+geom_main[5]*hand.shape[0],
		geom_main[0]+geom_main[1]*hand.shape[1],
		geom_main[3]]
#read river area
river_area = gdal.Open('elevation/buffered_rivers.tif').ReadAsArray()[1272:,1900:3399]
for dd in range(80):
	t_mid = t0+timedelta(days=dd)
	#load cloud free river part after VI
	arr=scipy.io.loadmat('Cloud_free/refl_vi'+t_mid.strftime('%Y%m%d')+'.mat')['refl_vi'+t_mid.strftime('%Y%m%d')]
	arr[:,river_area==-99] = 0
	#check if Landsat available
	##landsat bin file
	tm_files = glob('Landsat/Landsat_'+t_mid.strftime('%Y%m%d')+'_bin.tif')
	if tm_files:
		tm_file = tm_files[0]
		tm_resampled_file = 'validation/clipped_files/'+os.path.basename(tm_file)
		##ResampleImage
		ResampleImage(tm_file,tm_resampled_file,0.00219,bb)
		arr_tm = gdal.Open(tm_resampled_file).ReadAsArray()
		arr[:,arr_tm==1] = 1
	##assign the river part from the VI results
	mod_bin = arr[0,:,:]
	myd_bin = arr[1,:,:]
	#HAND threshold 19.5 meter
	HAND_water = 19.5
	#create water depth files
	mod_valid = np.where(np.logical_and(hand!=-99,mod_bin==1))
	mod_depth_arr = HAND_water - hand[mod_valid]
	mod_depth = np.ones(mod_bin.shape)*-99
	mod_depth[mod_valid] = mod_depth_arr
	mod_depth[mod_depth<0] = -99
	createTif('Water_depth/MOD_WD_'+t_mid.strftime('%Y%m%d')+'.tif',mod_depth,geom_main,gdal.GDT_Float32)
	myd_valid = np.where(np.logical_and(hand!=-99,myd_bin==1))
	myd_depth_arr = HAND_water - hand[myd_valid]
	myd_depth = np.ones(myd_bin.shape)*-99
	myd_depth[myd_valid] = myd_depth_arr
	myd_depth[myd_depth<0] = -99
	createTif('Water_depth/MYD_WD_'+t_mid.strftime('%Y%m%d')+'.tif',myd_depth,geom_main,gdal.GDT_Float32)
	
























