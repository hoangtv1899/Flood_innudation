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
plt.style.use('ggplot')
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

def train2(arr1, arr2, cloud_mask, water_mask,land_type=0, dem_arr = None):
	cloud_mask1 = cloud_mask==1
	temp = pd.DataFrame()
	val1 = []
	index1 = []
	type1 = []
	#b2-b1
	min1 = np.ones(arr1.shape)*-99
	min1[cloud_mask1] = arr2[cloud_mask1] - arr1[cloud_mask1]
	#ndvi
	ndvi = np.ones(arr1.shape)*-99
	sum1 = arr2 + arr1
	ndvi[cloud_mask1] = min1[cloud_mask1]/sum1[cloud_mask1].astype(np.float)
	#classify
	#band2
	#water
	a,b,c = classify('Band2',arr2,water_mask,1,'train')
	val1 += a; index1 += b; type1 += c
	#land
	a,b,c = classify('Band2',arr2,water_mask,land_type,'train')
	val1 += a; index1 += b; type1 += c
	#ndvi
	#water
	a,b,c = classify('ndvi',ndvi,water_mask,1,'train')
	val1 += a; index1 += b; type1 += c
	#land
	a,b,c = classify('ndvi',ndvi,water_mask,land_type,'train')
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

def test2(arr1, arr2, cloud_mask, dem_arr = None):
	cloud_mask1 = cloud_mask==1
	temp = pd.DataFrame()
	val1 = []
	index1 = []
	#b2-b1
	min1 = np.ones(arr1.shape)*-99
	min1[cloud_mask1] = arr2[cloud_mask1] - arr1[cloud_mask1]
	#ndvi
	ndvi = np.ones(arr1.shape)*-99
	sum1 = arr2 + arr1
	ndvi[cloud_mask1] = min1[cloud_mask1]/sum1[cloud_mask1].astype(np.float)
	#band2
	a,b = classify('Band2',arr2,cloud_mask,1)
	val1 += a; index1 += b
	#ndvi
	a,b = classify('ndvi',ndvi,cloud_mask,1)
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

def ClassifyMODIS(arr1,arr2,rf):
	cloud_mask = np.logical_and(arr1+arr2 !=0,
								np.logical_and(np.logical_and(arr2!=-28672,arr1!=-28672),
								np.logical_and(arr1!=0,arr1<2500))).astype(np.int)
	if len(np.unique(cloud_mask))==1:
		return
	MODClassTest = test(arr1, arr2, cloud_mask)
	#random forest predict
	features = list(MODClassTest.columns[:5])
	X2 = MODClassTest[features]
	y2_test = rf.predict(X2)
	res_arr = np.ones(arr1.shape)*-1
	try:
		res_arr[cloud_mask==1] = y2_test
	except:
		return
	return res_arr

def LearnMore(d1,d2,water_mask):
	#loop between days to modis files
	ndays = (d2-d1).days
	arr1 = np.array([]).reshape(0,water_mask.shape[0],water_mask.shape[1])
	arr2 = np.array([]).reshape(0,water_mask.shape[0],water_mask.shape[1])
	for i in range(ndays+1):
		currD = d1+timedelta(days=i)
		mFiles = sorted(glob('clipped_data/MOD09GQ.A'+currD.strftime('%Y%m%d')+'_b0[12].tif'))
		if len(mFiles) <2:
			continue
		temp_arr1 = gdal.Open(mFiles[0]).ReadAsArray()
		temp_arr2 = gdal.Open(mFiles[1]).ReadAsArray()
		arr1 = np.vstack([arr1,temp_arr1[np.newaxis,:,:]])
		arr2 = np.vstack([arr2,temp_arr2[np.newaxis,:,:]])
		
	cloud_mask = np.logical_and(arr2>0,
								np.logical_and(arr1<2500,arr1>0)).astype(np.int)
	if len(np.unique(cloud_mask))==1:
		return
	new_water_mask = np.repeat(water_mask[np.newaxis,:,:],ndays+1,axis=0)
	MODClassTrain = train(arr1, arr2, cloud_mask, new_water_mask)
	#random forest train
	y = MODClassTrain['Types']
	features = list(MODClassTrain.columns[:2])
	#features = ['Band2']
	X = MODClassTrain[features]
	class_weight = {1:0.7,0:0.3}
	dt = RandomForestClassifier(n_estimators=50,class_weight=class_weight,n_jobs=-1)
	dt.fit(X,y)
	joblib.dump(dt,'dt_2013.joblib.pkl',compress=9)
	return dt

"""
def WriteMODIS(d0,d1,refl_vi):
	t = d1.strftime('%Y%m%d')
	ndays, nr, nc = refl_vi.shape
	td = (d1-d0).days
	if td*2 > ndays:
		return
	file_oyscacld = glob('results/*_'+t+'*_bin.tif')[0]
	ds = gdal.Open(file_oyscacld)
	#file MOD
	mod_arr = refl_vi[td*2,:,:]
	createTif('Cloud_free/MOD_FM'+t+'.tif', mod_arr, ds.GetGeoTransform())
	#file MYD
	myd_arr = refl_vi[td*2+1,:,:]
	createTif('Cloud_free/MYD_FM'+t+'.tif', myd_arr, ds.GetGeoTransform())

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

def Landsat7Training(collected_sample, landsat_image):
	collected_sample_arr = gdal.Open(collected_sample).ReadAsArray()
	tm_files = sorted(glob(landsat_image+'*.[Tt][Ii][Ff]'))
	if not tm_files:
		return
	arr_red = gdal.Open(tm_files[0]).ReadAsArray().astype(np.float)
	arr_nir = gdal.Open(tm_files[1]).ReadAsArray().astype(np.float)
	arr_swir = gdal.Open(tm_files[2]).ReadAsArray().astype(np.float)
	legal_arr_tm = np.logical_and(np.logical_and(arr_red>1,arr_nir>1),#arr_red<11800)
									np.logical_and(arr_red<50,arr_swir>1))
	landsatClass = train(arr_red, arr_nir, legal_arr_tm, collected_sample_arr,0, arr_swir)
	y = landsatClass['Types']
	features = list(landsatClass.columns[:6])
	X = landsatClass[features]
	class_weight = {1:0.7,0:0.3}
	dt_tm = RandomForestClassifier(min_samples_split=10,class_weight=class_weight,random_state=99,n_jobs=-1)
	dt_tm.fit(X,y)
	return dt_tm

def MergeMODIS(arr1,arr2):
	res_arr = arr1.copy()
	more_w = np.logical_and(arr2==1, arr1<1)
	more_g = np.logical_and(arr2==0, arr1==-1)
	res_arr[more_w]=1
	res_arr[more_g]=0
	return res_arr

def Landsat8Classify(date0,dt_tm):
	landsat_images = sorted(glob('Landsat/LC08_LC01_*'+date0.strftime('%Y%m%d')+'*.[tT][iI][fF]'))
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
#	landsat_dem = gdal.Open('Landsat/landsat_dem.tif').ReadAsArray()
	res_arr_tm = np.ones(arr_red.shape)*-99
	res_arr_tm[legal_arr_tm] = y_test
#	res_arr_tm[np.logical_and(res_arr_tm==1,landsat_dem>=250)] = 0
	createTif('Landsat/Landsat_'+date0.strftime('%Y%m%d')+'_bin.tif', 
				res_arr_tm, 
				geom_tm, 
				gdal.GDT_Int16)

def Landsat7Classify(date0,dt_tm):
	landsat_images = sorted(glob('Landsat/LE07_LC01_*'+date0.strftime('%Y%m%d')+'*.[tT][iI][fF]'))
	if not landsat_images:
		return
	ds1_tm = gdal.Open(landsat_images[0])
	geom_tm = ds1_tm.GetGeoTransform()
	arr_red = ds1_tm.ReadAsArray().astype(np.float)
	arr_nir = gdal.Open(landsat_images[1]).ReadAsArray().astype(np.float)
	arr_swir = gdal.Open(landsat_images[2]).ReadAsArray().astype(np.float)
	legal_arr_tm = np.logical_and(np.logical_and(arr_red>1,arr_nir>1),#arr_red<11800)
									np.logical_and(arr_red<50,arr_swir>1))
	landsatClass = test(arr_red, arr_nir, legal_arr_tm, arr_swir)
	features1 = list(landsatClass.columns[:6])
	X_test = landsatClass[features1]
	y_test = dt_tm.predict(X_test)
	landsat_dem = gdal.Open('Landsat/landsat_dem.tif').ReadAsArray()
	res_arr_tm = np.ones(arr_red.shape)*-99
	res_arr_tm[legal_arr_tm] = y_test
	res_arr_tm[np.logical_and(res_arr_tm==1,landsat_dem>=250)] = 0
	createTif('Landsat/Landsat_'+date0.strftime('%Y%m%d')+'_bin.tif', 
				res_arr_tm, 
				geom_tm, 
				gdal.GDT_Int16)

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
	val_modf, type_modf, im_modf, res_modf = catValidation(arr_tm, arr_mod_flood, 'MWP',
														nan_arr_mod==0,w=2,l=2)
	date1 = [d0.strftime('%Y-%m-%d')]*14
	results['Date'] = date1
	results['Im_type'] = im_mod+im_modf
	results['Types'] = type_mod+type_modf
	results['Values'] = val_mod+val_modf
	return results, res_mod, res_modf
"""
def LandsatValidationEF5(d0,arr0,arr1,arr2):
	results = pd.DataFrame()
	nan_arr_mod = (arr0==-99).astype(np.int)
	val_cf, type_cf, im_cf, res_cf = catValidation(arr0, arr1, 'EF5',
														nan_arr_mod==0)
	val_mod, type_mod, im_mod, res_mod = catValidation(arr0, arr2, 'EF5_noncali',
														nan_arr_mod==0)
	date1 = [d0.strftime('%Y-%m-%d')]*14
	results['Date'] = date1
	results['Im_type'] = im_cf+im_mod
	results['Types'] = type_cf+type_mod
	results['Values'] = val_cf+val_mod
	return results, res_cf, res_mod


def ResampleImage(img,img1,resolution,bounding_box,srcnodata=-99):
	if srcnodata:
		os.system('gdalwarp -overwrite -r cubicspline -srcnodata '+str(srcnodata)+' -dstnodata -99 -tr '+\
					str(resolution)+' '+str(resolution)+\
					' -te '+' '.join([str(x) for x in bounding_box])+' '+img+' '+img1)
	else:
		os.system('gdalwarp -overwrite -r cubicspline -tr '+\
					str(resolution)+' '+str(resolution)+\
					' -te '+' '.join([str(x) for x in bounding_box])+' '+img+' '+img1)

def init(d0_, rf_):
	global d0, rf
	d0 = d0_
	rf = rf_

def ClassifyMultiple(i):
	d1 = d0+timedelta(days=i)
	mod_files = sorted(glob('clipped_data/MOD09GQ.A'+d1.strftime('%Y%m%d')+'_b0[12].tif'))
	if not mod_files or len(mod_files)<2:
		return
	if os.path.isfile('results/MOD_'+d1.strftime('%Y%m%d')+'_bin.tif'):
		return
	try:
		ds1 = gdal.Open(mod_files[0])
		geom = ds1.GetGeoTransform()
		mod_band1 = ds1.ReadAsArray()
		mod_band2 = gdal.Open(mod_files[1]).ReadAsArray()
		res_arr_mod0 = ClassifyMODIS(mod_band1,mod_band2,rf)
		createTif('results/MOD_'+d1.strftime('%Y%m%d')+'_bin.tif', res_arr_mod0, geom)
	except:
		print d1

"""
file_path = sys.argv[1]

if os.path.isdir(file_path+'reprojected/') == False:
	os.mkdir(file_path+'reprojected/')

if os.path.isdir(file_path+'clipped/') == False:
	os.mkdir(file_path+'clipped/')

#glob all file in the file path
headers = defaultdict(list)
all_files = sorted(glob(file_path+'*.*'))
for f in all_files:
	for header in ['LC08','MOD09GQ','MWP','MYD09GQ']:
		if ((header in f) and ('aux.xml' not in f)):
			headers[header].append(f)

ReprojectNClip(headers)
"""
#rf = joblib.load('rf7.joblib.pkl')
rf = joblib.load('rf_new7.joblib.pkl')
#LearnMore(date(2013,6,30),date(2013,6,30),
#				gdal.Open('elevation/swbd_mississippi_reprojected.tif').ReadAsArray())
"""
d0 = date(2012,5,18)
tm_path = 'Landsat/'
for i in range(1,130):
	d1 = d0+timedelta(days=i)
	dt_tm = Landsat7Training(tm_path+'collected_samples.tif',
			tm_path+'LE07_LC01_*'+d1.strftime("%Y%m%d"))
	Landsat7Classify(d1,dt_tm)

d0 = date(2013,4,1)

for i in range(1,130):
	d1 = d0+timedelta(days=i)
	dt_tm = Landsat8Training(tm_path+'collected_samples.tif',
			tm_path+'LC08_LC01_*'+d1.strftime("%Y%m%d"))
	Landsat8Classify(d1,dt_tm)
"""

d0 = date(2013,6,1)
p = mp.Pool()
p.map(ClassifyMultiple, range(50))
"""
d = date(2013,4,1)
temp_water_mask = np.array([]).reshape(0,1494,1836)
arr1_stack = np.array([]).reshape(0,1494,1836)
arr2_stack = np.array([]).reshape(0,1494,1836)
for i in range(90):
	d2 = d+timedelta(days=i)
	if os.path.isfile('results/MOD_'+d2.strftime('%Y%m%d')+'_bin.tif'):
		temp_arr = gdal.Open('results/MOD_'+d2.strftime('%Y%m%d')+'_bin.tif').ReadAsArray()
		temp_arr1 = gdal.Open('clipped_data/MOD09GQ.A'+d2.strftime('%Y%m%d')+'_b01.tif').ReadAsArray()
		temp_arr2 = gdal.Open('clipped_data/MOD09GQ.A'+d2.strftime('%Y%m%d')+'_b02.tif').ReadAsArray()
		temp_water_mask = np.vstack([temp_water_mask,temp_arr[np.newaxis,:,:]])
		arr1_stack = np.vstack([arr1_stack,temp_arr1[np.newaxis,:,:]])
		arr2_stack = np.vstack([arr2_stack,temp_arr2[np.newaxis,:,:]])

dt = LearnMore(arr1_stack,arr2_stack,temp_water_mask)
rf.estimators_ += dt.estimators_
rf.n_estimators = len(rf.estimators_)
arr1=gdal.Open('clipped_data/MOD09GQ.A20130630_b01.tif').ReadAsArray()
arr2=gdal.Open('clipped_data/MOD09GQ.A20130630_b02.tif').ReadAsArray()
new_res_arr = ClassifyMODIS(arr1,arr2,dt)
createTif('test1.tif',new_res_arr,geom)

"""
"""
d0 = date(2011,3,29)
dem_arr = gdal.Open('elevation/upstream_mississippi.tif').ReadAsArray()
tot_res = pd.DataFrame()
for i in range(300):
	d1 = d0+timedelta(days=i)
	#landsat bin file
	tm_files = glob('Landsat/Landsat_'+d1.strftime('%Y%m%d')+'_bin.tif')
	if not tm_files:
		continue
	tm_file = tm_files[0]
	tm_resampled_file = tm_file.split('.')[0]+'_resampled.tif'
#	ResampleLandsat(tm_file,tm_resampled_file)
	ds_tm = gdal.Open(tm_resampled_file)
	geom = ds_tm.GetGeoTransform()
	arr_tm = ds_tm.ReadAsArray()
	#modis classified bin file
	mod_files = sorted(glob('clipped_data/M[OY]D09GQ.A'+d1.strftime('%Y%m%d')+'_b0[12].tif'))
	if len(mod_files) == 4:
		mod_band1 = gdal.Open(mod_files[0]).ReadAsArray()
		mod_band2 = gdal.Open(mod_files[1]).ReadAsArray()
		myd_band1 = gdal.Open(mod_files[2]).ReadAsArray()
		myd_band2 = gdal.Open(mod_files[3]).ReadAsArray()
		res_arr_mod0 = ClassifyMODIS(mod_band1,mod_band2,dem_arr,rf)
		createTif('results/MOD_'+d1.strftime('%Y%m%d')+'_bin.tif',
					res_arr_mod0, geom)
		res_arr_mod1 = ReclassifyMODIS(mod_band1,mod_band2,res_arr_mod0,dem_arr,89)
		if res_arr_mod1 is not None:
			createTif('results/MOD_'+d1.strftime('%Y%m%d')+'_bin_new.tif',
					res_arr_mod1, geom)
		res_arr_myd0 = ClassifyMODIS(myd_band1,myd_band2,dem_arr,rf)
		createTif('results/MYD_'+d1.strftime('%Y%m%d')+'_bin.tif',
					res_arr_myd0, geom)
		res_arr_myd1 = ReclassifyMODIS(myd_band1,myd_band2,res_arr_myd0,dem_arr,89)
		if res_arr_myd1 is not None:
			createTif('results/MYD_'+d1.strftime('%Y%m%d')+'_bin_new.tif',
					res_arr_myd1, geom)
		if res_arr_mod1 is None:
			res_arr = MergeMODIS(res_arr_mod0,res_arr_myd1)
		elif res_arr_myd1 is None:
			res_arr = MergeMODIS(res_arr_mod1,res_arr_myd0)
		else:
			res_arr = MergeMODIS(res_arr_mod1,res_arr_myd1)
	elif len(mod_files) == 2:
		mod_band1 = gdal.Open(mod_files[0]).ReadAsArray()
		mod_band2 = gdal.Open(mod_files[1]).ReadAsArray()
		res_arr_mod0 = ClassifyMODIS(mod_band1,mod_band2,dem_arr,rf)
		res_arr = ReclassifyMODIS(mod_band1,mod_band2,res_arr_mod0,dem_arr,89)
	else:
		continue
#	createTif('results/MergeMODIS_'+d1.strftime('%Y%m%d')+'_bin.tif',
#					res_arr, geom)
	#mwp file
	mwp_file = glob('MWP/MWP_'+d1.strftime('%Y%j')+'_*.tif')
	if not mwp_file:
		pd_res, res_mod = LandsatValidation(d1,res_arr,arr_tm)
	else:
		arr_mod_flood = gdal.Open(mwp_file[0]).ReadAsArray()
		pd_res, res_mod,res_modf = LandsatValidation(d1,res_arr,arr_tm,arr_mod_flood)
	tot_res = tot_res.append(pd_res, ignore_index=True)

tot_res.to_csv('data.csv',index=False)
"""
#output from model
d0 = date(2013,6,14)
water_mask = gdal.Open('elevation/swbd_mississippi.tif').ReadAsArray()
tot_res = pd.DataFrame()
for d1 in [date(2013,6,7),date(2013,7,16),date(2013,7,25),
			date(2013,8,1),date(2013,8,10),date(2013,8,26),
			date(2014,7,28),date(2014,8,13),date(2014,8,29)]:
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
	
	#NASA flood map file
	mwp_file = glob('mwp/MWP_'+d1.strftime('%Y%j')+'*.tif')[0]
	clipped_mwp_file = 'validation/clipped_files/'+os.path.basename(mwp_file)
	ds_mwp = gdal.Open(mwp_file)
	geom_mwp = ds_mwp.GetGeoTransform()
	arr_mwp = ds_mwp.ReadAsArray()
	bb_mwp = [geom_mwp[0],geom_mwp[3]-arr_mwp.shape[0]*geom_mwp[1],
				geom_mwp[0]+arr_mwp.shape[1]*geom_mwp[1],geom_mwp[3]]
	#final bb
	bb = [max(bb_tm[0],bb_mwp[0]),max(bb_tm[1],bb_mwp[1]),
			min(bb_tm[2],bb_mwp[2]),min(bb_tm[3],bb_mwp[3])]
	
	#ResampleLandsat
	ResampleImage(tm_file,tm_resampled_file,0.00219,bb_tm)
	ResampleImage(mwp_file,clipped_mwp_file,0.00219,bb,None)
	#modis cloud free file
	ds_mod = gdal.Open('results/MOD_'+d1.strftime('%Y%m%d')+'_bindem.tif')
	geom_mod = ds_mod.GetGeoTransform()
	cloud_free_arr = scipy.io.loadmat('Cloud_free/refl_vi'+d1.strftime('%Y%m%d')+'.mat')['refl_vi'+d1.strftime('%Y%m%d')]
	cloud_free_file = 'Cloud_free/MOD_CF_'+d1.strftime('%Y%m%d')+'.tif'
	createTif(cloud_free_file,cloud_free_arr,geom_mod)
	cloud_free_file_clipped = 'validation/clipped_files/'+os.path.basename(cloud_free_file)
	ResampleImage(cloud_free_file,cloud_free_file_clipped,0.00219,bb)
	#model file
#	model_file = glob('ef5/mississippi_flood_map/output/depth.'+
#						d1.strftime('%Y%m%d')+'_*.tif')[0]
#	arr_model = gdal.Open(model_file).ReadAsArray()[44:-463,128:-637]
#	arr_model[np.logical_and(water_mask<1,arr_model<8)] = 0
#	arr_model[arr_model>0]=1
#	arr_model[arr_model<0]=0
#	model_file_mod = 'ef5/mississippi_flood_map/output/mod_'+os.path.basename(model_file)
#	model_file_clipped = 'validation/clipped_files/'+os.path.basename(model_file)
#	createTif(model_file_mod,arr_model,geom_mod)
#	ResampleImage(model_file_mod,model_file_clipped,0.005,bb_tm)
	#model file no calibration
#	model_file1 = glob('ef5/mississippi_flood_map/output1/depth.'+
#						d1.strftime('%Y%m%d')+'_*.tif')[0]
#	arr_model1 = gdal.Open(model_file1).ReadAsArray()[44:-463,128:-637]
#	arr_model1[arr_model1>0.5]=1
#	arr_model1[arr_model1<0]=0
#	model_file_mod1 = 'ef5/mississippi_flood_map/output1/mod_'+os.path.basename(model_file)
#	model_file_clipped1 = 'validation/clipped_files/'+os.path.basename(model_file).split('.tif')[0]+'01.tif'
#	createTif(model_file_mod1,arr_model1,geom_mod)
#	ResampleImage(model_file_mod1,model_file_clipped1,0.005,bb_tm)
	tm_ds = gdal.Open('validation/clipped_files/Landsat_'+d1.strftime('%Y%m%d')+'_bin.tif')
	geom = tm_ds.GetGeoTransform()
	tm_val = tm_ds.ReadAsArray()
	#cf_val = gdal.Open('validation/clipped_files/MOD_CF_'+d1.strftime('%Y%m%d')+'.tif').ReadAsArray()
	#mwp_val = gdal.Open(glob('validation/clipped_files/MWP_'+d1.strftime('%Y%j')+'*.tif')[0]).ReadAsArray()
#	mod_val = gdal.Open('validation/clipped_files/depth.'+d1.strftime('%Y%m%d')+'_0000.simpleinundation.tif').ReadAsArray()
#	mod_val1 = gdal.Open('validation/clipped_files/depth.'+d1.strftime('%Y%m%d')+'_0000.simpleinundation01.tif').ReadAsArray()
#	pd_res, res_cf, res_mod = LandsatValidationEF5(d1,tm_val,mod_val,mod_val1)
	#createTif('validation/CF_'+d1.strftime('%Y%m%d')+'_cat.tif',res_cf,geom)
	#createTif('validation/MWP_'+d1.strftime('%Y%m%d')+'_cat.tif',res_mwp,geom)
	mod_val = gdal.Open(cloud_free_file_clipped).ReadAsArray()
	modf_val = gdal.Open(clipped_mwp_file).ReadAsArray()
	pd_res, res_mod, res_modf = LandsatValidation(d1,mod_val,tm_val,modf_val)
	res_modf[modf_val==0]=-1
	createTif('validation/MWP_'+d1.strftime('%Y%m%d')+'_cat.tif',res_modf,geom)
	createTif('validation/CF_'+d1.strftime('%Y%m%d')+'_cat.tif',res_mod,geom)
	tot_res = tot_res.append(pd_res, ignore_index=True)

tot_res.to_csv('model_vs_modis4.csv',index=False)

#train model
tot_res = pd.DataFrame()
for d1 in [date(2013,6,7),date(2013,7,16),date(2013,7,25),
			date(2013,8,1),date(2013,8,10),date(2013,8,26),
			date(2014,7,28),date(2014,8,13),date(2014,8,29)]:
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
	
	#ResampleLandsat
	ResampleImage(tm_file,tm_resampled_file,0.005,bb_tm)
	#modis classified file
	ds1 = gdal.Open('clipped_data/MOD09GQ.A'+d1.strftime('%Y%m%d')+'_b01.tif')
	geom_mod = ds1.GetGeoTransform()
	mod_arr1 = ds1.ReadAsArray()
	mod_arr2 = gdal.Open('clipped_data/MOD09GQ.A'+d1.strftime('%Y%m%d')+'_b02.tif').ReadAsArray()
	new_res_arr = ClassifyMODIS(mod_arr1,mod_arr2,dt)
	mod_class_file = 'results/MOD_'+d1.strftime('%Y%m%d')+'_bin.tif'
	createTif(mod_class_file,new_res_arr,geom_mod)
	mod_class_file_clipped = 'validation/clipped_files/'+os.path.basename(mod_class_file)
	ResampleImage(mod_class_file,mod_class_file_clipped,0.005,bb_tm)
	
	tm_ds = gdal.Open('validation/clipped_files/Landsat_'+d1.strftime('%Y%m%d')+'_bin.tif')
	geom = tm_ds.GetGeoTransform()
	tm_val = tm_ds.ReadAsArray()
	
	mod_val = gdal.Open(mod_class_file_clipped).ReadAsArray()
	
	pd_res = pd.DataFrame()
	nan_arr_mod = (tm_val==-99).astype(np.int)
	val_cf, type_cf, im_cf, res_cf = catValidation(tm_val, mod_val, 'MOD_Classified',
														nan_arr_mod==0)
	date1 = [d1.strftime('%Y-%m-%d')]*7
	pd_res['Date'] = date1
	pd_res['Im_type'] = im_cf
	pd_res['Types'] = type_cf
	pd_res['Values'] = val_cf
	
	createTif('validation/MOD_'+d1.strftime('%Y%m%d')+'_cat.tif',res_cf,geom)
	tot_res = tot_res.append(pd_res, ignore_index=True)

#Learn More
water_mask = np.array([]).reshape(0,350,224)
arr1 = np.array([]).reshape(0,350,224)
arr2 = np.array([]).reshape(0,350,224)

for currD in [date(2014,7,28),date(2014,8,13),date(2014,8,29)]:
	ds_water_mask = gdal.Open('validation/MOD_'+currD.strftime('%Y%m%d')+'_cat.tif')
	geom_water_mask = ds_water_mask.GetGeoTransform()
	temp_water_mask = ds_water_mask.ReadAsArray()
	tm_val = gdal.Open('validation/clipped_files/Landsat_'+d1.strftime('%Y%m%d')+'_bin.tif').ReadAsArray()
	bb_cat = [geom_water_mask[0],geom_water_mask[3]-temp_water_mask.shape[0]*geom_water_mask[1],
				geom_water_mask[0]+temp_water_mask.shape[1]*geom_water_mask[1],geom_water_mask[3]]
	
	mFiles = sorted(glob('clipped_data/MOD09GQ.A'+currD.strftime('%Y%m%d')+'_b0[12].tif'))
	for jk,mFile in enumerate(mFiles):
		mFile_clipped = 'validation/clipped_files/'+os.path.basename(mFile)
		ResampleImage(mFile,mFile_clipped,0.005,bb_cat)
		temp_arr = gdal.Open(mFile_clipped).ReadAsArray()
		if jk == 0:
			arr1 = np.vstack([arr1,temp_arr[np.newaxis,:,:]])
		elif jk == 1:
			arr2 = np.vstack([arr2,temp_arr[np.newaxis,:,:]])
	
	if (len(np.unique(temp_water_mask)) < 4) or (np.sum(temp_water_mask==3)< 30):
		temp_water_mask[temp_water_mask!=2] = -99
		temp_water_mask[temp_water_mask==2] = 1
		temp_water_mask[tm_val==0] = 0
	else:
		temp_water_mask[temp_water_mask<2] = -99
		temp_water_mask[temp_water_mask==2] = 1
		temp_water_mask[temp_water_mask==3] = 0
	water_mask = np.vstack([water_mask,temp_water_mask[np.newaxis,:,:]])

cloud_mask = np.logical_and(arr2>0,
							np.logical_and(arr1<2500,arr1>0)).astype(np.int)
MODClassTrain = train(arr1, arr2, cloud_mask, water_mask)
#random forest train
y = MODClassTrain['Types']
features = list(MODClassTrain.columns[:5])
#features = ['Band2']
X = MODClassTrain[features]
class_weight = {1:0.7,0:0.3}
dt = RandomForestClassifier(n_estimators=50,class_weight=class_weight,n_jobs=-1)
dt.fit(X,y)

rf.estimators_ += dt.estimators_
rf.n_estimators = len(rf.estimators_)
joblib.dump(rf,'rf7_updated.joblib.pkl',compress=9)

#set threshold for water
t0 = date(2014,7,21)
#t0 = date(2014,7,21)
list_days = [(t0+timedelta(days=i)).strftime('%Y%m%d') for i in range(47)]
#list_days = [(t0+timedelta(days=i)).strftime('%Y%m%d') for i in range(45)]
#dem_arr = gdal.Open('elevation/dem_new.tif').ReadAsArray()
arr0 = gdal.Open('clipped_data/MOD09GQ.A20140803_b01.tif').ReadAsArray()
geom = gdal.Open('clipped_data/MOD09GQ.A20140803_b01.tif').GetGeoTransform()
lons = np.tile(np.linspace(geom[0],geom[0]+geom[1]*arr0.shape[1],arr0.shape[1]),\
																(arr0.shape[0],1))
lats = np.tile(np.linspace(geom[3],geom[3]-geom[1]*arr0.shape[0],arr0.shape[0]).reshape(-1,1),\
																(1,arr0.shape[1]))

arrG = 360 + np.arctan(lons/lats)*180/np.pi
A = 13.5
B = 1081.1
C = 0.7
D = 2027.0
E = 675.7
for d in list_days:
	for pre in ['MOD','MYD']:
		#read data
		arr1 = gdal.Open('clipped_data/'+pre+'09GQ.A'+d+'_b01.tif').ReadAsArray()
		arr2 = gdal.Open('clipped_data/'+pre+'09GQ.A'+d+'_b02.tif').ReadAsArray()
		arr3 = gdal.Open('clipped_data/'+pre+'09GA.A'+d+'_b03.tif').ReadAsArray()
		arrEV = gdal.Open('clipped_data/'+pre+'09GA.A'+d+'_SensorZenith.tif').ReadAsArray()
		arrEV = np.ma.masked_where(arrEV < -18000, arrEV)*0.01
		arrES = gdal.Open('clipped_data/'+pre+'09GA.A'+d+'_SolarZenith.tif').ReadAsArray()
		arrES = np.ma.masked_where(arrES < -18000, arrES)*0.01
		arrBV = gdal.Open('clipped_data/'+pre+'09GA.A'+d+'_SensorAzimuth.tif').ReadAsArray()
		arrBV = np.ma.masked_where(arrBV < -18000, arrBV)*0.01
		arrBS = gdal.Open('clipped_data/'+pre+'09GA.A'+d+'_SolarAzimuth.tif').ReadAsArray()
		arrBS = np.ma.masked_where(arrBS < -18000, arrBS)*0.01
		#mask cloud
		cloud_mask = arr1>=D
		#water classification
		res_arr = (np.divide((arr2+A),(arr1+B)) < C).astype(np.int)
		res_arr[cloud_mask] = -1
		#mask cloud shadow
		x_cloud, y_cloud = np.where(cloud_mask)
		x_shifts = np.tan(np.pi*arrEV[cloud_mask]/180.)*\
					np.sin(np.pi*(arrBV[cloud_mask]+arrG[cloud_mask])/180.) - \
					np.tan(np.pi*arrES[cloud_mask]/180.)*\
					np.sin(np.pi*(arrBS[cloud_mask]+arrG[cloud_mask])/180.)
		y_shifts = np.tan(np.pi*arrEV[cloud_mask]/180.)*\
					np.cos(np.pi*(arrBV[cloud_mask]+arrG[cloud_mask])/180.) - \
					np.tan(np.pi*arrES[cloud_mask]/180.)*\
					np.cos(np.pi*(arrBS[cloud_mask]+arrG[cloud_mask])/180.)
		cloud = cloud_mask.astype(np.int)
		for hc in np.arange(0.5,12.5,0.5):
			x_shadows = x_cloud + np.around(hc*x_shifts).astype(np.int)
			y_shadows = y_cloud + np.around(hc*y_shifts).astype(np.int)
			x_valid = np.logical_and(x_shadows>0,x_shadows<arr1.shape[0]-1)
			y_valid = np.logical_and(y_shadows>0,y_shadows<arr1.shape[1]-1)
			x_shadows = x_shadows[np.logical_and(x_valid,y_valid)]
			y_shadows = y_shadows[np.logical_and(x_valid,y_valid)]
			cloud[x_shadows,y_shadows] = 1
		#remove cloud shadow falsely classified as water
		#ratio between band 2 and band 3
		arr23 = arr2/(arr3).astype(np.float)
		res_arr[np.logical_and(arr23>=1.5,np.logical_and(cloud==1,res_arr==1))] = 0
		#final classified flood maps
		createTif('results/'+pre+'_'+d+'_bindem.tif',res_arr,geom)
		





