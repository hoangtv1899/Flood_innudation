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
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import multiprocessing as mp
import sys
from sklearn.externals import joblib

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

def train(arr1, arr2, cloud_mask, water_mask,land_type=0, arr3 = None, dem_arr = None):
	cloud_mask1 = cloud_mask==1
	temp = pd.DataFrame()
	val1 = []
	index1 = []
	type1 = []
	#b2-b1
	min1 = np.ones(arr1.shape)*-99
	min1[cloud_mask1] = arr2[cloud_mask1] - arr1[cloud_mask1]
	#b2/b1
	rat1 = np.ones(arr1.shape)*-99
	rat1[cloud_mask1] = arr2[cloud_mask1]/arr1[cloud_mask1].astype(np.float)
	#ndvi
	ndvi = np.ones(arr1.shape)*-99
	sum1 = arr2 + arr1
	ndvi[cloud_mask1] = min1[cloud_mask1]/sum1[cloud_mask1].astype(np.float)
	if arr3 is not None:
		#ndwi
		ndwi = np.ones(arr1.shape)*-9999
		rat_tm2 = np.ones(arr1.shape)*-9999
		rat_tm2[cloud_mask1] = arr3[cloud_mask1] / arr2[cloud_mask1]
		ndwi[cloud_mask1] = (np.ones(arr1.shape)[cloud_mask1]-rat_tm2[cloud_mask1])/(np.ones(arr1.shape)[cloud_mask1]+rat_tm2[cloud_mask1])
	#classify
	#band1
	#water
	a,b,c = classify('Band1',arr1,water_mask,1,'train')
	val1 += a; index1 += b; type1 += c
	#land
	a,b,c = classify('Band1',arr1,water_mask,land_type,'train')
	val1 += a; index1 += b; type1 += c
	#band2
	#water
	a,b,c = classify('Band2',arr2,water_mask,1,'train')
	val1 += a; index1 += b; type1 += c
	#land
	a,b,c = classify('Band2',arr2,water_mask,land_type,'train')
	val1 += a; index1 += b; type1 += c
	#band2 - band1
	#water
	a,b,c = classify('Band2 - Band1',min1,water_mask,1,'train')
	val1 += a; index1 += b; type1 += c
	#land
	a,b,c = classify('Band2 - Band1',min1,water_mask,land_type,'train')
	val1 += a; index1 += b; type1 += c
	#b2/b1
	#water
	a,b,c = classify('Band2 / Band1',rat1,water_mask,1,'train')
	val1 += a; index1 += b; type1 += c
	#land
	a,b,c = classify('Band2 / Band1',rat1,water_mask,land_type,'train')
	val1 += a; index1 += b; type1 += c
	#ndvi
	#water
	a,b,c = classify('ndvi',ndvi,water_mask,1,'train')
	val1 += a; index1 += b; type1 += c
	#land
	a,b,c = classify('ndvi',ndvi,water_mask,land_type,'train')
	val1 += a; index1 += b; type1 += c
	if arr3 is not None:
		#ndwi
		#water
		a,b,c = classify('ndwi',ndwi,water_mask,1,'train')
		val1 += a; index1 += b; type1 += c
		#land
		a,b,c = classify('ndwi',ndwi,water_mask,land_type,'train')
		val1 += a; index1 += b; type1 += c
	if dem_arr is not None:
		#dem
		#water
		a,b,c = classify('dem',dem_arr,water_mask,1,'train')
		val1 += a; index1 += b; type1 += c
		#land
		a,b,c = classify('dem',dem_arr,water_mask,land_type,'train')
		val1 += a; index1 += b; type1 += c
	temp['Val'] = val1
	temp['Index'] = index1
	temp['Types'] = type1
	df_mod_train = pd.DataFrame()
	for idx in temp['Index'].unique().tolist():
		df_mod_train[idx] = temp.loc[temp['Index']==idx]['Val'].reset_index(drop=True)
	df_mod_train['Types'] = temp.loc[temp['Index']==idx]['Types'].reset_index(drop=True)
	df_mod_train = df_mod_train.dropna()
	return df_mod_train

def test(arr1, arr2, cloud_mask, arr3 = None, dem_arr = None):
	cloud_mask1 = cloud_mask==1
	temp = pd.DataFrame()
	val1 = []
	index1 = []
	#b2-b1
	min1 = np.ones(arr1.shape)*-99
	min1[cloud_mask1] = arr2[cloud_mask1] - arr1[cloud_mask1]
	#b2/b1
	rat1 = np.ones(arr1.shape)*-99
	rat1[cloud_mask1] = arr2[cloud_mask1]/arr1[cloud_mask1].astype(np.float)
	#ndvi
	ndvi = np.ones(arr1.shape)*-99
	sum1 = arr2 + arr1
	ndvi[cloud_mask1] = min1[cloud_mask1]/sum1[cloud_mask1].astype(np.float)
	if arr3 is not None:
		#ndwi
		ndwi = np.ones(arr1.shape)*-9999
		rat_tm2 = np.ones(arr1.shape)*-9999
		rat_tm2[cloud_mask1] = arr3[cloud_mask1] / arr2[cloud_mask1]
		ndwi[cloud_mask1] = (np.ones(arr1.shape)[cloud_mask1]-rat_tm2[cloud_mask1])/(np.ones(arr1.shape)[cloud_mask1]+rat_tm2[cloud_mask1])
	#band1
	a,b = classify('Band1',arr1,cloud_mask,1)
	val1 += a; index1 += b
	#band2
	a,b = classify('Band2',arr2,cloud_mask,1)
	val1 += a; index1 += b
	#band2 - band1
	a,b = classify('Band2 - Band1',min1,cloud_mask,1)
	val1 += a; index1 += b
	#b2/b1
	a,b = classify('Band2 / Band1',rat1,cloud_mask,1)
	val1 += a; index1 += b
	#ndvi
	a,b = classify('ndvi',ndvi,cloud_mask,1)
	val1 += a; index1 += b
	if arr3 is not None:
		#ndwi
		a,b = classify('ndwi',ndwi,cloud_mask,1)
		val1 += a; index1 += b
	if dem_arr is not None:
		#dem
		a,b = classify('dem',dem_arr,cloud_mask,1)
		val1 += a; index1 += b
	temp['Val'] = val1
	temp['Index'] = index1
	df_mod_test = pd.DataFrame()
	for idx in temp['Index'].unique().tolist():
		df_mod_test[idx] = temp.loc[temp['Index']==idx]['Val'].reset_index(drop=True)
	df_mod_test = df_mod_test.dropna()
	return df_mod_test

def classify(index,arr,mask,type,train=None):
	select_arr = arr[mask==type]
	val1 = select_arr.tolist()
	index1 = [index]*len(select_arr)
	if train == 'train':
		if type%2 == 1:
			type1 = np.ones((1,len(select_arr))).tolist()[0]
		elif type%2 == 0:
			type1 = np.zeros((1,len(select_arr))).tolist()[0]
		return val1, index1, type1
	else:
		return val1, index1

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

def Landsat8Training(collected_sample, landsat_image):
	collected_sample_arr = gdal.Open(collected_sample).ReadAsArray()
	tm_files = sorted(glob(landsat_image+'*.[Tt][Ii][Ff]'))
	if not tm_files:
		return
	arr_red = gdal.Open(tm_files[0]).ReadAsArray().astype(np.float)
	arr_nir = gdal.Open(tm_files[1]).ReadAsArray().astype(np.float)
	arr_swir = gdal.Open(tm_files[2]).ReadAsArray().astype(np.float)
	legal_arr_tm = np.logical_and(np.logical_and(arr_red>1,arr_nir>1),#arr_red<11800)
									np.logical_and(arr_red<10000,arr_swir>1))
	landsatClass = train(arr_red, arr_nir, legal_arr_tm, collected_sample_arr,0, arr_swir)
	y = landsatClass['Types']
	features = list(landsatClass.columns[:6])
	X = landsatClass[features]
	class_weight = {1:0.7,0:0.3}
	dt_tm = RandomForestClassifier(min_samples_split=10,class_weight=class_weight,random_state=99,n_jobs=-1)
	dt_tm.fit(X,y)
	return dt_tm

def Landsat8Classify(date0,dt_tm):
	landsat_images = sorted(glob('Landsat/LC08_L1TP_*'+date0.strftime('%Y%m%d')+'*.[tT][iI][fF]'))
	if not landsat_images:
		return
	ds1_tm = gdal.Open(landsat_images[0])
	geom_tm = ds1_tm.GetGeoTransform()
	arr_red = ds1_tm.ReadAsArray().astype(np.float)
	arr_nir = gdal.Open(landsat_images[1]).ReadAsArray().astype(np.float)
	arr_swir = gdal.Open(landsat_images[2]).ReadAsArray().astype(np.float)
	legal_arr_tm = np.logical_and(np.logical_and(arr_red>1,arr_nir>1),#arr_red<11800)
									np.logical_and(arr_red<18000,arr_swir>1))
	landsatClass = test(arr_red, arr_nir, legal_arr_tm, arr_swir)
	features1 = list(landsatClass.columns[:6])
	X_test = landsatClass[features1]
	y_test = dt_tm.predict(X_test)
	res_arr_tm = np.ones(arr_red.shape)*-99
	res_arr_tm[legal_arr_tm] = y_test
	createTif('Landsat/Landsat_'+date0.strftime('%Y%m%d')+'_bin.tif', 
				res_arr_tm, 
				geom_tm, 
				gdal.GDT_Int16)

def ResampleLandsat(img,img1,resample_method,resolution):
	os.system('gdalwarp -overwrite -r '+resample_method+' -srcnodata -99 -dstnodata -99 -tr '+resolution+' '+resolution+' '+img+' '+img1)

def LandsatValidation(d0,arr_mod,arr_tm,arr_mod_flood=None):
	results = pd.DataFrame()
	if arr_mod_flood is None:
		nan_arr_mod = np.logical_or(arr_mod==-1,arr_tm==-99).astype(np.int)
		val_mod, type_mod, im_mod, res_mod = catValidation(arr_tm, arr_mod, 'Model',
														nan_arr_mod==0)
		results['Date'] = [d0.strftime('%Y-%m-%d')]*7
		results['Im_type'] = im_mod
		results['Types'] = type_mod
		results['Values'] = val_mod
		return results, res_mod
	nan_arr_mod = np.logical_or(np.logical_or(arr_mod==-1,arr_mod_flood==0),arr_tm==-99).astype(np.int)
	val_mod, type_mod, im_mod, res_mod = catValidation(arr_tm, arr_mod, 'MOD',
														nan_arr_mod==0)
	#for mod flood images
	val_modf, type_modf, im_modf, res_modf = catValidation(arr_tm, arr_mod_flood, 'MWP',
														nan_arr_mod==0,w=2,l=2)
	date1 = [d0.strftime('%Y-%m-%d')]*14
	results['Date'] = date1
	results['Im_type'] = im_mod+im_modf
	results['Types'] = type_mod+type_modf
	results['Values'] = val_mod+val_modf
	return results, res_mod, res_modf

t_start = datetime(2017,4,1)
#Landsat 8
fileProjected8 = 'ref_satellite/reprojected/LC08_20170510_bin_reprojected.tif'
dateLandsat8 = datetime.strptime(os.path.basename(fileProjected8).split('_')[1],'%Y%m%d')
fileResampled8 = fileProjected8.split('.tif')[0]+'_resampled.tif'

#ResampleLandsat(fileProjected8,fileResampled8,'cubic','750')
arr_tm8 = gdal.Open(fileResampled8).ReadAsArray()
arr_mod0 = np.loadtxt('results/res_SGC-%04d.wd' %(dateLandsat8-t_start).days , skiprows=6)
result8, res_mod8 = LandsatValidation(dateLandsat8, (arr_mod0>8).astype(np.int), arr_tm8)
result8

#Landsat 7
fileProjected7 = 'ref_satellite/reprojected/LE07_20170502_bin_reprojected.tif'
dateLandsat7 = datetime.strptime(os.path.basename(fileProjected7).split('_')[1],'%Y%m%d')
fileResampled7 = fileProjected7.split('.tif')[0]+'_resampled.tif'
#ResampleLandsat(fileProjected7,fileResampled7,'cubic','750')
arr_tm7 = gdal.Open(fileResampled7).ReadAsArray()
arr_mod1 = np.loadtxt('results/res_SGC-%04d.wd' %(dateLandsat7-t_start).days , skiprows=6)
result7, res_mod7 = LandsatValidation(dateLandsat7, (arr_mod1>8).astype(np.int), arr_tm7)
result7












