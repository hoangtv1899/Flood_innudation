#!/data/apps/anaconda/2.7-4.3.1/bin/python

import scipy.io
import os
import numpy as np
from osgeo import gdal, gdalnumeric, ogr
from datetime import date, datetime, timedelta
from MODIS_floodmap import GetMOD09GQ, GetMOD09GA
from sklearn.externals import joblib
from glob import glob
import scipy.ndimage
import pandas as pd
from sklearn.tree import DecisionTreeClassifier#, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import multiprocessing as mp
from multiprocessing import Manager
from contextlib import closing

#GetMOD09GQ(date(2011,4,23), date(2011,5,17), ['h11v04'])

water_mask = gdal.Open('/ssd-scratch/htranvie/Flood/data/elevation/SWBD_mississippi_resampled3.tif').ReadAsArray()
d0 = date(2000,2,24)

for i in range(367):
	t = (d0+timedelta(days=i)).strftime('%Y%m%d')
	for pre in ['MOD']:
		temp = pd.DataFrame()
		val = []
		types = []
		index = []
		header = '/ssd-scratch/htranvie/Flood/data/clipped_data/'+pre+'09GQ.A'+t
		try:
			arr1 = gdal.Open(header+'_b01.tif').ReadAsArray()
			arr2 = gdal.Open(header+'_b02.tif').ReadAsArray()
		except:
			continue
		#cloud and nodata mask
		cloud_mask = np.logical_and(arr1+arr2 !=0,
									np.logical_and(np.logical_and(arr2!=-28672,arr1!=-28672),
									np.logical_and(arr1!=0,arr1<2000))).astype(np.int)
		if len(np.unique(cloud_mask)) ==1:
			continue
		#b2-b1
		min1 = np.ones(arr1.shape)*-99
		min1[cloud_mask==1] = arr2[cloud_mask==1] - arr1[cloud_mask==1]
		#b2/b1
		rat1 = np.ones(arr1.shape)*-99
		rat1[cloud_mask==1] = arr2[cloud_mask==1]/arr1[cloud_mask==1].astype(np.float)
		#ndvi
		ndvi = np.ones(arr1.shape)*-99
		sum1 = arr2 + arr1
		ndvi[cloud_mask==1] = min1[cloud_mask==1]/sum1[cloud_mask==1].astype(np.float)
		#classify
		#water band1
		pix_wa_b1 = arr1[np.logical_and(water_mask==1,cloud_mask==1)]
		val += pix_wa_b1.tolist()
		types += ['Water']*len(pix_wa_b1)
		index += ['Band1']*len(pix_wa_b1)
		#land band1
		pix_land_b1 = arr1[np.logical_and(water_mask==0,cloud_mask==1)]
		val += pix_land_b1.tolist()
		types += ['Land']*len(pix_land_b1)
		index += ['Band1']*len(pix_land_b1)
		#water band2
		pix_wa_b2 = arr2[np.logical_and(water_mask==1,cloud_mask==1)]
		val += pix_wa_b2.tolist()
		types += ['Water']*len(pix_wa_b2)
		index += ['Band2']*len(pix_wa_b2)
		#land band2
		pix_land_b2 = arr2[np.logical_and(water_mask==0,cloud_mask==1)]
		val += pix_land_b2.tolist()
		types += ['Land']*len(pix_land_b2)
		index += ['Band2']*len(pix_land_b2)
		#water band2 - band1
		pix_wa_min = min1[np.logical_and(water_mask==1,cloud_mask==1)]
		val += pix_wa_min.tolist()
		types += ['Water']*len(pix_wa_min)
		index += ['Band2 - Band1']*len(pix_wa_min)
		#land band2 - band1
		pix_land_min = min1[np.logical_and(water_mask==0,cloud_mask==1)]
		val += pix_land_min.tolist()
		types += ['Land']*len(pix_land_min)
		index += ['Band2 - Band1']*len(pix_land_min)
		#water b2/b1
		pix_wa_rat = rat1[np.logical_and(water_mask==1,cloud_mask==1)]
		val += pix_wa_rat.tolist()
		types += ['Water']*len(pix_wa_rat)
		index += ['Band2 / Band1']*len(pix_wa_rat)
		#land b2/b1
		pix_land_rat = rat1[np.logical_and(water_mask==0,cloud_mask==1)]
		val += pix_land_rat.tolist()
		types += ['Land']*len(pix_land_rat)
		index += ['Band2 / Band1']*len(pix_land_rat)
		#water ndvi
		pix_wa_ndvi = ndvi[np.logical_and(water_mask==1,cloud_mask==1)]
		val += pix_wa_ndvi.tolist()
		types += ['Water']*len(pix_wa_ndvi)
		index += ['ndvi']*len(pix_wa_ndvi)
		#land ndvi
		pix_land_ndvi = ndvi[np.logical_and(water_mask==0,cloud_mask==1)]
		val += pix_land_ndvi.tolist()
		types += ['Land']*len(pix_land_ndvi)
		index += ['ndvi']*len(pix_land_ndvi)
		temp['Val'] = val
		temp['Types'] = types
		temp['Index'] = index
		temp.to_csv('/ssd-scratch/htranvie/Flood/data/csv/new_hit_'+pre+t+'.csv')

csv_2000 = sorted(glob('/ssd-scratch/htranvie/Flood/data/csv/new_hit_MOD2000*.csv'))+\
			sorted(glob('/ssd-scratch/htranvie/Flood/data/csv/new_hit_MOD2001*.csv'))
train_data = pd.DataFrame()
for csv_file in csv_2000:
	temp_train_data = pd.read_csv(csv_file)
	train_data = train_data.append(temp_train_data, ignore_index=True)

df_mod = train_data.copy()
df_mod1 = pd.DataFrame()
for idx in df_mod['Index'].unique().tolist():
	df_mod1[idx] = df_mod.loc[df_mod['Index']==idx]['Val'].reset_index(drop=True)

df_mod1['Types'] = df_mod.loc[df_mod['Index']==idx]['Types'].reset_index(drop=True)
df_mod1 = df_mod1.dropna()
types = df_mod1['Types'].tolist()
int_types = [1 if x=='Water' else 0 for x in types]
df_mod1['Targets'] = int_types

y = df_mod1['Targets']
features = list(df_mod1.columns[:5])
X = df_mod1[features]
class_weight = {1:0.6,0:0.4}
#decision tree
dt = RandomForestClassifier(min_samples_split=100,class_weight=class_weight,random_state=99,n_jobs=-1)
dt.fit(X,y)
#joblib.dump(dt,'/ssd-scratch/htranvie/Flood/data/rf.joblib.pkl',compress=9)
#dt = joblib.load('/ssd-scratch/htranvie/Flood/data/rf.joblib.pkl')

#test dataset
d0 = date(2008,5,28)

for i in range(35):
	t = (d0+timedelta(days=i)).strftime('%Y%m%d')
	for pre in ['MOD','MYD']:
		temp = pd.DataFrame()
		val = []
		index = []
		header = '/ssd-scratch/htranvie/Flood/data/clipped_data/'+pre+'09GQ.A'+t
		try:
			arr1 = gdal.Open(header+'_b01.tif').ReadAsArray()
			arr2 = gdal.Open(header+'_b02.tif').ReadAsArray()
		except:
			continue
		cloud_mask = np.logical_and(arr1+arr2 !=0,
									np.logical_and(np.logical_and(arr2!=-28672,arr1!=-28672),
									np.logical_and(arr1!=0,arr1<1500))).astype(np.int)
		if len(np.unique(cloud_mask))==1:
			continue
		#b2-b1
		min1 = np.ones(arr1.shape)*-99
		min1[cloud_mask==1] = arr2[cloud_mask==1] - arr1[cloud_mask==1]
		#b2/b1
		rat1 = np.ones(arr1.shape)*-99
		rat1[cloud_mask==1] = arr2[cloud_mask==1]/arr1[cloud_mask==1].astype(np.float)
		#ndvi
		ndvi = np.ones(arr1.shape)*-99
		sum1 = arr2 + arr1
		ndvi[cloud_mask==1] = min1[cloud_mask==1]/sum1[cloud_mask==1].astype(np.float)
		#classify
		#band1
		pix_land_b1 = arr1[cloud_mask==1]
		val += pix_land_b1.tolist()
		index += ['Band1']*len(pix_land_b1)
		#band2
		pix_land_b2 = arr2[cloud_mask==1]
		val += pix_land_b2.tolist()
		index += ['Band2']*len(pix_land_b2)
		#band2 - band1
		pix_land_min = min1[cloud_mask==1]
		val += pix_land_min.tolist()
		index += ['Band2 - Band1']*len(pix_land_min)
		#b2/b1
		pix_land_rat = rat1[cloud_mask==1]
		val += pix_land_rat.tolist()
		index += ['Band2 / Band1']*len(pix_land_rat)
		#ndvi
		pix_land_ndvi = ndvi[cloud_mask==1]
		val += pix_land_ndvi.tolist()
		index += ['ndvi']*len(pix_land_ndvi)
		temp['Val'] = val
		temp['Index'] = index
		temp.to_csv('/ssd-scratch/htranvie/Flood/data/csv/new_hit_'+pre+t+'.csv')

#check the result
d0 = date(2008,5,28)
miss_dem = gdal.Open('/ssd-scratch/htranvie/Flood/data/elevation/mississippi_elevation_clipped2.tif').ReadAsArray()
for i in range(35):
	for pre in ['MOD','MYD']:
		t = (d0+timedelta(days=i)).strftime("%Y%m%d")
		try:
			test_data = pd.read_csv('/ssd-scratch/htranvie/Flood/data/csv/new_hit_'+pre+t+'.csv')
		except:
			continue
		df_mod2 = pd.DataFrame()
		for idx in test_data['Index'].unique().tolist():
			df_mod2[idx] = test_data.loc[test_data['Index']==idx]['Val'].reset_index(drop=True)
		df_mod2 = df_mod2.dropna()
		#decision tree predict
		X2 = df_mod2[features]
		y2_test = dt.predict(X2)
		header = '/ssd-scratch/htranvie/Flood/data/clipped_data/'+pre+'09GQ.A'+t
		ds1 = gdal.Open(header+'_b01.tif')
		arr1 = ds1.ReadAsArray()
		geom = ds1.GetGeoTransform()
		arr2 = gdal.Open(header+'_b02.tif').ReadAsArray()
		cloud_mask = np.logical_and(arr1+arr2 !=0,
									np.logical_and(np.logical_and(arr2!=-28672,arr1!=-28672),
									np.logical_and(arr1!=0,arr1<1500))).astype(np.int)
		res_arr = np.ones(arr1.shape)*-1
		try:
			res_arr[cloud_mask==1] = y2_test
		except:
			print header
			continue
		res_arr[np.logical_and(miss_dem>172,res_arr==1)]=0
		res_arr[np.logical_and(np.logical_and(cloud_mask==1,water_mask==1),res_arr==0)]=1
		driver = gdal.GetDriverByName('GTiff')
		#owl
		dataset = driver.Create(
				'/ssd-scratch/htranvie/Flood/data/results/'+pre+'_'+t+'_bin.tif',
				res_arr.shape[1],
				res_arr.shape[0],
				1,
				gdal.GDT_Int16,
				)
		dataset.SetGeoTransform(geom)
		outband = dataset.GetRasterBand(1)
		outband.WriteArray(res_arr)
		outband.FlushCache()
		outband.SetNoDataValue(-99)
		dataset.FlushCache()

#Landsat validation
d_tm = date(2008,6,1)
for d_tm in [date(2017,4,7)]:
	collected_samples = gdal.Open('/ssd-scratch/htranvie/Flood/data/landsat/test_data/collected_samples_'+d_tm.strftime('%Y%m%d')+'.tif').ReadAsArray()
	temp = pd.DataFrame()
	val = []
	index = []
	types = []
	temp1 = pd.DataFrame()
	val1 = []
	index1 = []
	ds1_tm = gdal.Open('/ssd-scratch/htranvie/Flood/data/landsat/test_data/L7_'+d_tm.strftime('%Y%j')+'_b1.tif')
	geom_tm = ds1_tm.GetGeoTransform()
	arr_tm1 = ds1_tm.ReadAsArray().astype(np.float)
	arr_tm2 = gdal.Open('/ssd-scratch/htranvie/Flood/data/landsat/test_data/L7_'+d_tm.strftime('%Y%j')+'_b2.tif').ReadAsArray().astype(np.float)
	legal_arr_tm = np.logical_and(arr_tm1>1,arr_tm2>1)
	#b2-b1
	min_tm1 = np.ones(arr_tm1.shape)*-9999
	min_tm1[legal_arr_tm] = arr_tm2[legal_arr_tm] - arr_tm1[legal_arr_tm]
	#b2/b1
	rat_tm1 = np.ones(arr_tm1.shape)*-9999
	rat_tm1[legal_arr_tm] = arr_tm2[legal_arr_tm]/(arr_tm1[legal_arr_tm]).astype(np.float)
	#ndvi
	ndvi_tm = np.ones(arr_tm1.shape)*-9999
	sum_tm1 = np.ones(arr_tm1.shape)*-9999
	sum_tm1[legal_arr_tm] = arr_tm2[legal_arr_tm] + arr_tm1[legal_arr_tm]
	ndvi_tm[legal_arr_tm] = min_tm1[legal_arr_tm]/sum_tm1[legal_arr_tm].astype(np.float)
	#classify
	#band1
	#water
	pix_wa_b1 = arr_tm1[np.logical_and(legal_arr_tm,collected_samples==1)]
	val1 += arr_tm1[legal_arr_tm].tolist()
	index1 += ['Band1']*np.sum(legal_arr_tm)
	val += pix_wa_b1.tolist()
	index += ['Band1']*len(pix_wa_b1)
	types += ['Water']*len(pix_wa_b1)
	#land
	pix_land_b1 = arr_tm1[np.logical_and(legal_arr_tm,collected_samples==0)]
	val += pix_land_b1.tolist()
	index += ['Band1']*len(pix_land_b1)
	types += ['Land']*len(pix_land_b1)
	#cloud
	pix_cloud_b1 = arr_tm1[np.logical_and(legal_arr_tm,collected_samples==2)]
	val += pix_cloud_b1.tolist()
	index += ['Band1']*len(pix_cloud_b1)
	types += ['Cloud']*len(pix_cloud_b1)
	#band2
	#water
	pix_wa_b2 = arr_tm2[np.logical_and(legal_arr_tm,collected_samples==1)]
	val1 += arr_tm2[legal_arr_tm].tolist()
	index1 += ['Band2']*np.sum(legal_arr_tm)
	val += pix_wa_b2.tolist()
	index += ['Band2']*len(pix_wa_b2)
	types += ['Water']*len(pix_wa_b2)
	#land
	pix_land_b2 = arr_tm2[np.logical_and(legal_arr_tm,collected_samples==0)]
	val += pix_land_b2.tolist()
	index += ['Band2']*len(pix_land_b2)
	types += ['Land']*len(pix_land_b2)
	#cloud
	pix_cloud_b2 = arr_tm2[np.logical_and(legal_arr_tm,collected_samples==2)]
	val += pix_cloud_b2.tolist()
	index += ['Band2']*len(pix_cloud_b2)
	types += ['Cloud']*len(pix_cloud_b2)
	#band2 - band1
	#water
	pix_wa_min = min_tm1[np.logical_and(legal_arr_tm,collected_samples==1)]
	val1 += min_tm1[legal_arr_tm].tolist()
	index1 += ['Band2 - Band1']*np.sum(legal_arr_tm)
	val += pix_wa_min.tolist()
	index += ['Band2 - Band1']*len(pix_wa_min)
	types += ['Water']*len(pix_wa_min)
	#land
	pix_land_min = min_tm1[np.logical_and(legal_arr_tm,collected_samples==0)]
	val += pix_land_min.tolist()
	index += ['Band2 - Band1']*len(pix_land_min)
	types += ['Land']*len(pix_land_min)
	#cloud
	pix_cloud_min = min_tm1[np.logical_and(legal_arr_tm,collected_samples==2)]
	val += pix_cloud_min.tolist()
	index += ['Band2 - Band1']*len(pix_cloud_min)
	types += ['Cloud']*len(pix_cloud_min)
	#b2/b1
	#water
	pix_wa_rat = rat_tm1[np.logical_and(legal_arr_tm,collected_samples==1)]
	val1 += rat_tm1[legal_arr_tm].tolist()
	index1 += ['Band2 / Band1']*np.sum(legal_arr_tm)
	val += pix_wa_rat.tolist()
	index += ['Band2 / Band1']*len(pix_wa_rat)
	types += ['Water']*len(pix_wa_rat)
	#land
	pix_land_rat = rat_tm1[np.logical_and(legal_arr_tm,collected_samples==0)]
	val += pix_land_rat.tolist()
	index += ['Band2 / Band1']*len(pix_land_rat)
	types += ['Land']*len(pix_land_rat)
	#cloud
	pix_cloud_rat = rat_tm1[np.logical_and(legal_arr_tm,collected_samples==2)]
	val += pix_cloud_rat.tolist()
	index += ['Band2 / Band1']*len(pix_cloud_rat)
	types += ['Cloud']*len(pix_cloud_rat)
	#ndvi
	#water
	pix_wa_ndvi = ndvi_tm[np.logical_and(legal_arr_tm,collected_samples==1)]
	val1 += ndvi_tm[legal_arr_tm].tolist()
	index1 += ['ndvi']*np.sum(legal_arr_tm)
	val += pix_wa_ndvi.tolist()
	index += ['ndvi']*len(pix_wa_ndvi)
	types += ['Water']*len(pix_wa_ndvi)
	#land
	pix_land_ndvi = ndvi_tm[np.logical_and(legal_arr_tm,collected_samples==0)]
	val += pix_land_ndvi.tolist()
	index += ['ndvi']*len(pix_land_ndvi)
	types += ['Land']*len(pix_land_ndvi)
	#cloud
	pix_cloud_ndvi = ndvi_tm[np.logical_and(legal_arr_tm,collected_samples==2)]
	val += pix_cloud_ndvi.tolist()
	index += ['ndvi']*len(pix_cloud_ndvi)
	types += ['Cloud']*len(pix_cloud_ndvi)
	temp['Val'] = val
	temp['Index'] = index
	temp['Types'] = types
	temp.to_csv('/ssd-scratch/htranvie/Flood/data/csv/Landsat_'+d_tm.strftime('%Y%j')+'_train.csv')
	temp1['Val'] = val1
	temp1['Index'] = index1
	temp1.to_csv('/ssd-scratch/htranvie/Flood/data/csv/Landsat_'+d_tm.strftime('%Y%j')+'_test.csv')
	
	df_mod = temp.copy()
	df_mod1 = pd.DataFrame()
	for idx in df_mod['Index'].unique().tolist():
		df_mod1[idx] = df_mod.loc[df_mod['Index']==idx]['Val'].reset_index(drop=True)
	
	df_mod1['Types'] = df_mod.loc[df_mod['Index']==idx]['Types'].reset_index(drop=True)
	df_mod1 = df_mod1.dropna()
	types = df_mod1['Types'].tolist()
	int_types = []
	for x in types:
		if x =='Water':
			int_types.append(1)
		elif x == 'Land':
			int_types.append(0)
		elif x == 'Cloud':
			int_types.append(2)
	df_mod1['Targets'] = int_types
	
	y = df_mod1['Targets']
	features = list(df_mod1.columns[:5])
	X = df_mod1[features]
	class_weight = {1:0.5,0:0.3,2:0.2}
	#decision tree
	dt_tm = RandomForestClassifier(min_samples_split=300,class_weight=class_weight,random_state=99,n_jobs=-1)
	dt_tm.fit(X,y)
	#joblib.dump(dt_tm,'/ssd-scratch/htranvie/Flood/data/rf_tm.joblib.pkl',compress=9)
	#dt_tm = joblib.load('/ssd-scratch/htranvie/Flood/data/rf_tm.joblib.pkl')
	df_mod0 = temp1.copy()
	df_mod01 = pd.DataFrame()
	for idx in df_mod0['Index'].unique().tolist():
		df_mod01[idx] = df_mod0.loc[df_mod0['Index']==idx]['Val'].reset_index(drop=True)
	
	df_mod01 = df_mod01.dropna()
	
	features1 = list(df_mod01.columns[:5])
	X_test = df_mod01[features1]
	y_test = dt_tm.predict(X_test)
	res_arr_tm = np.ones(arr_tm1.shape)*-99
	res_arr_tm[legal_arr_tm] = y_test
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(
			'/ssd-scratch/htranvie/Flood/data/results/Landsat_'+d_tm.strftime('%Y%m%d')+'_bin.tif',
			res_arr_tm.shape[1],
			res_arr_tm.shape[0],
			1,
			gdal.GDT_Int16,
			)
	dataset.SetGeoTransform(geom_tm)
	outband = dataset.GetRasterBand(1)
	outband.WriteArray(res_arr_tm)
	outband.FlushCache()
	outband.SetNoDataValue(-99)
	dataset.FlushCache()

#upscale Landsat to MODIS resolution
land1 = gdal.Open('/ssd-scratch/htranvie/Flood/data/landsat/test_data/land.tif').ReadAsArray()
#land2 = gdal.Open('/ssd-scratch/htranvie/Flood/data/landsat/test_data/land1.tif').ReadAsArray()
#land3 = gdal.Open('/ssd-scratch/htranvie/Flood/data/landsat/test_data/land2.tif').ReadAsArray()
for d_tm in [date(2017,4,23)]:
#	tm_check_file = '/ssd-scratch/htranvie/Flood/data/landsat/test_data/L7_'+d_tm.strftime('%Y%j')+'_b1.tif'
#	tm_check_resample = '/ssd-scratch/htranvie/Flood/data/landsat/test_data/L7_'+d_tm.strftime('%Y%j')+'_resampled1.tif'
	mod_file = '/ssd-scratch/htranvie/Flood/data/results/MOD_'+d_tm.strftime('%Y%m%d')+'_bin.tif'
	mod_resample_file = '/ssd-scratch/htranvie/Flood/data/results/MOD_'+d_tm.strftime('%Y%m%d')+'_resampled1.tif'
	tm_file = '/ssd-scratch/htranvie/Flood/data/results/Landsat_'+d_tm.strftime('%Y%m%d')+'_bin.tif'
	tm_resample_file = '/ssd-scratch/htranvie/Flood/data/results/Landsat_'+d_tm.strftime('%Y%m%d')+'_resampled1.tif'
	
#	os.system('gdalwarp -overwrite -dstnodata -99 -ts 116 127 '+tm_check_file+' '+tm_check_resample)
	os.system('gdalwarp -overwrite -r cubicspline -srcnodata -99 -dstnodata -99 -ts 116 127 '+tm_file+' '+tm_resample_file)
#	os.system('gdalwarp -overwrite -r cubicspline -ts 1684 1837 '+mod_file+' '+mod_resample_file)
	ds_mod = gdal.Open(mod_file)
	arr_mod = ds_mod.ReadAsArray()
	arr_tm = gdal.Open(tm_resample_file).ReadAsArray()
#	tm_check = gdal.Open(tm_check_resample).ReadAsArray()
	arr_tm[land1==0]=0
	arr_mod[land1==0]=0
	arr_tm[land1==2]=1
	arr_mod[land1==1]=1
	
	#hit
	hit_arr = np.logical_and(arr_mod==1,arr_tm==1)
	no_hit = np.sum(hit_arr)
	#miss
	miss_arr = np.logical_and(arr_mod==0,arr_tm==1)
	no_miss = np.sum(miss_arr)
	#false
	false_arr = np.logical_and(arr_mod==1,arr_tm==0)
	no_false = np.sum(false_arr)
	#correct negative
	corrneg = np.logical_and(arr_mod==0,arr_tm==0)
	no_corrneg = np.sum(corrneg)
	no_hit,no_miss,no_false,no_corrneg
	no_hit/(float(no_miss)+no_hit), no_false/(float(no_hit)+no_false)
	#result array
	res_arr = hit_arr+miss_arr*2+false_arr*3
	driver = gdal.GetDriverByName('GTiff')
	dataset1 = driver.Create(
			'/ssd-scratch/htranvie/Flood/data/results/MOD_'+d_tm.strftime('%Y%m%d')+'_cat.tif',
			res_arr.shape[1],
			res_arr.shape[0],
			1,
			gdal.GDT_Int16,
			)
	dataset1.SetGeoTransform(ds_mod.GetGeoTransform())
	outband1 = dataset1.GetRasterBand(1)
	outband1.WriteArray(res_arr)
	outband1.FlushCache()
	outband1.SetNoDataValue(0)
	dataset1.FlushCache()
	
	



L = scipy.ndimage.zoom(refl_in,(1,2,2),order=1)

refl_vi = ClearCloud(L, date(2017,4,1), date(2017,4,15))
scipy.io.savemat('/ssd-scratch/htranvie/Flood/data/refl_vi.mat', mdict={'refl_vi':refl_vi})
