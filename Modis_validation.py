#!/data/apps/anaconda/2.7-4.3.1/bin/python

import scipy.io
import os
import numpy as np
from osgeo import gdal, gdalnumeric, ogr, osr
from datetime import date, datetime, timedelta
from glob import glob
import scipy.ndimage
import pandas as pd
from collections import defaultdict
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import multiprocessing as mp
import sys

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

def catValidation(ref_arr, mod_arr, type_name, mask, w=1, l=0):
	val = []
	type = []
	im_type = []
	#hit
	if w == 1:
		hit_arr = np.logical_and(mod_arr[mask]==1,ref_arr[mask]==1)
	else:
		hit_arr = np.logical_and(mod_arr[mask]>=2,ref_arr[mask]==1)
	no_hit = np.sum(hit_arr)
	#miss
	if l==0:
		miss_arr = np.logical_and(mod_arr[mask]==0,ref_arr[mask]==1)
	elif l==1:
		miss_arr = np.logical_and(mod_arr[mask]<=0,ref_arr[mask]==1)
	elif l==2:
		miss_arr = np.logical_and(mod_arr[mask]==1,ref_arr[mask]==1)
	no_miss = np.sum(miss_arr)
	#false
	if w==1:
		false_arr = np.logical_and(mod_arr[mask]==w,ref_arr[mask]==0)
	else:
		false_arr = np.logical_and(mod_arr[mask]>=2,ref_arr[mask]==0)
	no_false = np.sum(false_arr)
	#correct negative
	if l==0:
		corrneg = np.logical_and(mod_arr[mask]==0,ref_arr[mask]==0)
	elif l==1:
		corrneg = np.logical_and(mod_arr[mask]<=0,ref_arr[mask]==0)
	elif l==2:
		corrneg = np.logical_and(mod_arr[mask]==1,ref_arr[mask]==0)
	no_corrneg = np.sum(corrneg)
	val += [no_hit,no_miss,no_false,no_corrneg]
	type += ['Hit','Miss','False','Correct Negative']
	val += [no_hit/(float(no_miss)+no_hit), no_false/(float(no_hit)+no_false), (no_hit*no_corrneg-no_false*no_miss)/float((no_hit+no_miss)*(no_corrneg+no_false))]
	type += ['POD','FAR','HK']
	im_type += [type_name]*7
	res_arr = np.zeros(mod_arr.shape)
	res_arr[mask] = hit_arr+miss_arr*2+false_arr*3
	return val, type, im_type, res_arr

def LandsatValidation(d0,arr_mod,arr_tm,arr_mod_flood=None):
	results = pd.DataFrame()
	if arr_mod_flood is None:
		nan_arr_mod = np.logical_or(arr_mod==-1,arr_tm==-99).astype(np.int)
		val_mod, type_mod, im_mod, res_mod = catValidation(arr_tm, arr_mod, 'MOD',
														nan_arr_mod==0)
		results['Date'] = [d0.strftime('%Y-%m-%d')]*7
		results['Im_type'] = im_mod
		results['Types'] = type_mod
		results['Values'] = val_mod
		return results, res_mod
	nan_arr_mod = (arr_tm==-99).astype(np.int)
	val_mod, type_mod, im_mod, res_mod = catValidation(arr_tm, arr_mod, 'MOD',
														nan_arr_mod==0)
	#for mod flood images
	val_modf, type_modf, im_modf, res_modf = catValidation(arr_tm, arr_mod_flood, 'MYD',
														nan_arr_mod==0)
	date1 = [d0.strftime('%Y-%m-%d')]*14
	results['Date'] = date1
	results['Im_type'] = im_mod+im_modf
	results['Types'] = type_mod+type_modf
	results['Values'] = val_mod+val_modf
	return results, res_mod, res_modf

def ResampleImage(img,img1,resolution,bounding_box,srcnodata=-99):
	if srcnodata:
		os.system('gdalwarp -overwrite -r cubicspline -srcnodata '+str(srcnodata)+' -dstnodata -99 -tr '+\
					str(resolution)+' '+str(resolution)+\
					' -te '+' '.join([str(x) for x in bounding_box])+' '+img+' '+img1)
	else:
		os.system('gdalwarp -overwrite -r cubicspline -tr '+\
					str(resolution)+' '+str(resolution)+\
					' -te '+' '.join([str(x) for x in bounding_box])+' '+img+' '+img1)



tot_res = pd.DataFrame()
for d1 in [date(2013,6,14),date(2013,6,30),date(2013,7,16),date(2013,7,25),date(2013,8,1),
			date(2013,8,10),date(2013,8,26)]:
	#landsat bin file
	tm_files = glob('Landsat/Landsat_'+d1.strftime('%Y%m%d')+'_bin.tif')
	if not tm_files:
		continue
	tm_file = tm_files[0]
	tm_resampled_file = 'validation/clipped_files/'+os.path.basename(tm_file)
	ds_tm = gdal.Open(tm_file)
	geom_tm = ds_tm.GetGeoTransform()
	arr_tm = ds_tm.ReadAsArray()
	bb_tm = [geom_tm[0],geom_tm[3]-arr_tm.shape[0]*geom_tm[1],
				geom_tm[0]+arr_tm.shape[1]*geom_tm[1],geom_tm[3]]
	#modis cloud-free file
	cf_file = 'Cloud_free/refl_vi'+d1.strftime('%Y%m%d')+'.mat'
	if not os.path.isfile(cf_file):
		continue
	arr_cf = scipy.io.loadmat(cf_file)['refl_vi'+d1.strftime('%Y%m%d')]
	mod_cf = arr_cf[0,:,:]
	myd_cf = arr_cf[1,:,:]
	geom_cf = (-93.20055786027632, 0.00219, 0.0, 44.995849995506845, 0.0, -0.00219)
	#add new information from new classified files
	arr_mod_bin = gdal.Open('results_test/MOD_'+d1.strftime('%Y%m%d')+'_bindem.tif').ReadAsArray()[1272:,1900:3399]
	arr_myd_bin = gdal.Open('results_test/MYD_'+d1.strftime('%Y%m%d')+'_bindem.tif').ReadAsArray()[1272:,1900:3399]
	river_area = gdal.Open('elevation/buffered_rivers.tif').ReadAsArray()[1272:,1900:3399]
	mod_cf[np.logical_and(mod_cf==0,arr_mod_bin==1)] = 1
	mod_cf[np.logical_and(mod_cf==1,river_area==-99)] = 0
	myd_cf[np.logical_and(myd_cf==0,arr_myd_bin==1)] = 1
	myd_cf[np.logical_and(myd_cf==1,river_area==-99)] = 0
	mod_cf_file = 'validation/clipped_files/MOD_CF_'+d1.strftime('%Y%m%d')+'.tif'
	myd_cf_file = 'validation/clipped_files/MYD_CF_'+d1.strftime('%Y%m%d')+'.tif'
	mod_cf_file_resample = mod_cf_file.split('.')[0]+'_resampled.tif'
	myd_cf_file_resample = myd_cf_file.split('.')[0]+'_resampled.tif'
	createTif(mod_cf_file, mod_cf, geom_cf)
	createTif(myd_cf_file, myd_cf, geom_cf)
	bb_cf = [geom_cf[0],geom_cf[3]-mod_cf.shape[0]*geom_cf[1],
				geom_cf[0]+mod_cf.shape[1]*geom_cf[1],geom_cf[3]]
	#final bb
	bb = [max(bb_tm[0],bb_cf[0]),max(bb_tm[1],bb_cf[1]),
			min(bb_tm[2],bb_cf[2]),min(bb_tm[3],bb_cf[3])]
	#ResampleImage
	ResampleImage(tm_file,tm_resampled_file,0.00219,bb)
	ResampleImage(mod_cf_file,mod_cf_file_resample,0.00219,bb)
	ResampleImage(myd_cf_file,myd_cf_file_resample,0.00219,bb)
	#Read final images
	arr_tm = gdal.Open(tm_resampled_file).ReadAsArray()
	arr_mod = gdal.Open(mod_cf_file_resample).ReadAsArray()
	arr_myd = gdal.Open(myd_cf_file_resample).ReadAsArray()
	geom_final = gdal.Open(myd_cf_file_resample).GetGeoTransform()
	#Validation
	pd_res, res_mod,res_myd = LandsatValidation(d1,arr_mod,arr_tm,arr_myd)
	createTif('validation/MOD_CF_'+d1.strftime('%Y%m%d')+'_cat.tif',res_mod,geom_final)
	createTif('validation/MYD_CF_'+d1.strftime('%Y%m%d')+'_cat.tif',res_myd,geom_final)
	tot_res = tot_res.append(pd_res, ignore_index=True)

tot_res.to_csv('data_0216_new.csv',index=False)
