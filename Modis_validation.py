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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import multiprocessing as mp
import sys
from sklearn.externals import joblib

def ReprojectNClip(headers):
	ds0 = gdal.Open(headers['MWP'][0])
	ulx0, xres0, xskew0, uly0, yskew0, yres0 = ds0.GetGeoTransform()
	lrx0 = ulx0 + ds0.RasterXSize*xres0
	lry0 = uly0 + ds0.RasterYSize*yres0
	extend = [str(ulx0), str(uly0), str(lrx0), str(lry0)]
	for header in ['LC08','MOD09GQ','MYD09GQ','MWP']:
		# first reproject files to epsg:4326
		hfiles = headers[header]
		if header == 'LC08':
			ds = gdal.Open(hfiles[0])
			prj = ds.GetProjection()
			srs = osr.SpatialReference(wkt=prj)
			epsg = srs.GetAttrValue("AUTHORITY",1)
			for hfile in hfiles:
				rpj_file = file_path+'reprojected/'+os.path.basename(hfile)
				if not os.path.isfile(rpj_file):
					cmd1 = 'gdalwarp -overwrite -s_srs EPSG:'+epsg+' -t_srs EPSG:4326 -r cubic '+hfile+' '+rpj_file
					os.system(cmd1)
				ds0 = gdal.Open(rpj_file)
				ulx, xres, xskew, uly, yskew, yres = ds0.GetGeoTransform()
				lrx = ulx + ds0.RasterXSize*xres
				lry = uly + ds0.RasterYSize*yres
				a0 = np.array([ulx0,uly0,lrx0,lry0])
				a_new = np.array([ulx,uly,lrx,lry])
				compare_extend = np.array([ulx0<ulx,uly0>uly,lrx0>lrx,lry0<lry])
				#compare between landsat and mwp
				if all(compare_extend):
					extend = [str(ulx), str(uly), str(lrx), str(lry)]
					os.system('cp '+rpj_file+' '+file_path+'clipped/.')
				else:
					a_new[compare_extend==False] = a0[compare_extend==False]
					extend = [str(x) for x in a_new.tolist()]
					clipped_file = file_path+'clipped/'+os.path.basename(hfile)
					if not os.path.isfile(clipped_file):
						cmd2 = 'gdal_translate -projwin '+' '.join(extend)+' '+rpj_file+' '+clipped_file
						os.system(cmd2)
		elif header in ['MOD09GQ','MYD09GQ']:
			tr = 0.00219684
			for hfile in hfiles:
				rpj_file = file_path+'reprojected/'+os.path.basename(hfile)
				clipped_file = file_path+'clipped/'+os.path.basename(hfile)
				hname, hext = os.path.splitext(hfile)
				if hext == '.tif':
					if not os.path.isfile(rpj_file):
						cmd3 = 'gdalwarp -overwrite -t_srs "+proj=longlat +datum=WGS84 +no_defs" -dstnodata -28672 -r cubic -tr '+str(tr)+' '+str(tr)+' -ot Int16 -of GTiff '+hfile+' '+rpj_file
						os.system(cmd3)
					#clip to landsat extend
					if not os.path.isfile(clipped_file):
						cmd4 = 'gdal_translate -projwin '+' '.join(extend)+' '+rpj_file+' '+clipped_file
						os.system(cmd4)
				elif hext == '.hdf':
					for b in ['b01','b02']:
						dt0 = datetime.strptime(file_path[:-1].split('/')[-1],'%Y-%m-%d')
						hdf_files_b = 'HDF4_EOS:EOS_GRID:'+hfile+':MODIS_Grid_2D:sur_refl_'+b+'_1'
						#moisac hdf files
						vrt_file = file_path+hname.split('/')[-1]+b+'.vrt'
						if not os.path.isfile(vrt_file):
							os.system('gdalbuildvrt '+vrt_file+' '+hdf_files_b)
						#convert and reproject vrt file to geotiff
						tif_file = file_path+header+'.A'+dt0.strftime('%Y%j')+'_'+b+'.tif'
						if not os.path.isfile(tif_file):
							cmd5 = 'gdalwarp -overwrite -t_srs "+proj=longlat +datum=WGS84 +no_defs" -dstnodata -28672 -r cubic -tr '+str(tr)+' '+str(tr)+' -ot Int16 -of GTiff '+vrt_file+' '+tif_file
							os.system(cmd5)
						clipped_file1 = file_path+'clipped/'+os.path.basename(tif_file)
						if not os.path.isfile(clipped_file1):
							cmd6 = 'gdal_translate -projwin '+' '.join(extend)+' '+tif_file+' '+clipped_file1
							os.system(cmd6)
		else:
			#tr = 0.00219684
			#only need to clip the MWP file
			for hfile in hfiles:
				clipped_file2 = file_path+'clipped/'+os.path.basename(hfile)
				if not os.path.isfile(clipped_file2):
					cmd7 = 'gdal_translate -projwin '+' '.join(extend)+' '+hfile+' '+clipped_file2
					os.system(cmd7)
	
	with open(file_path+'clipped/extend.txt','w') as fo:
		fo.write(' '.join(extend))

def ClassifyMODIS(d0,i,pre):
	t = (d0+timedelta(days=i)).strftime('%Y%m%d')
	temp = pd.DataFrame()
	val = []
	index = []
	header = '/ssd-scratch/htranvie/Flood/data/clipped_data/'+pre+'09GQ.A'+t
	#header = '/ssd-scratch/htranvie/Flood/data/downloads/'+(d0+timedelta(days=i)).strftime('%Y-%m-%d')+'/'+pre+'09GQ.A'+(d0+timedelta(days=i)).strftime('%Y%j')
	try:
		arr1 = gdal.Open(header+'_b01.tif').ReadAsArray()
		arr2 = gdal.Open(header+'_b02.tif').ReadAsArray()
	except:
		return
	cloud_mask = np.logical_and(arr1+arr2 !=0,
								np.logical_and(np.logical_and(arr2!=-28672,arr1!=-28672),
								np.logical_and(arr1!=0,arr1<1700))).astype(np.int)
	if len(np.unique(cloud_mask))==1:
		return
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
#		temp.to_csv('/ssd-scratch/htranvie/Flood/data/csv/new_hit_'+pre+t+'.csv',index=False)
	test_data = temp.copy()
	df_mod2 = pd.DataFrame()
	for idx in test_data['Index'].unique().tolist():
		df_mod2[idx] = test_data.loc[test_data['Index']==idx]['Val'].reset_index(drop=True)
	df_mod2 = df_mod2.dropna()
	#decision tree predict
	features = list(df_mod2.columns[:5])
	X2 = df_mod2[features]
	y2_test = dt.predict(X2)
	ds1 = gdal.Open(header+'_b01.tif')
	arr1 = ds1.ReadAsArray()
	geom = ds1.GetGeoTransform()
	arr2 = gdal.Open(header+'_b02.tif').ReadAsArray()
	res_arr = np.ones(arr1.shape)*-1
	try:
		res_arr[cloud_mask==1] = y2_test
	except:
		print header
		return
#		res_arr[np.logical_and(miss_dem>215,res_arr==1)]=0
#		res_arr[np.logical_and(np.logical_and(cloud_mask==1,water_mask1==1),res_arr==0)]=1
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

def ReclassifyMODIS(d0,pre):
	t = d0.strftime('%Y%m%d')
	temp = pd.DataFrame()
	val = []
	index = []
	temp1 = pd.DataFrame()
	val1 = []
	index1 = []
	type1 = []
	header = '/ssd-scratch/htranvie/Flood/data/downloads/'+d0.strftime('%Y-%m-%d')+'/'+pre+'09GQ.A'+d0.strftime('%Y%j')
	try:
		arr1 = gdal.Open(header+'_b01.tif').ReadAsArray()
		arr2 = gdal.Open(header+'_b02.tif').ReadAsArray()
	except:
		return
	water_mask = gdal.Open('/ssd-scratch/htranvie/Flood/data/results/'+pre+'_'+t+'_bin.tif').ReadAsArray()
	n_pix_water = np.sum(water_mask==1)
	#random select land pixels
	xx,yy = np.where(np.logical_and(water_mask==0,np.logical_and(arr2>3000,arr2<5000)))
	pix_land = np.random.randint(0,xx.shape[0],n_pix_water).tolist()
	for pix in pix_land:
		water_mask[xx[pix],yy[pix]] = 2
	cloud_mask = np.logical_and(arr1+arr2 !=0,
								np.logical_and(np.logical_and(arr2!=-28672,arr1!=-28672),
								np.logical_and(arr1!=0,arr1<1700))).astype(np.int)
	if len(np.unique(cloud_mask))==1:
		return
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
	#water
	pix_wa_b1 = arr1[water_mask==1]
	val1 += pix_wa_b1.tolist()
	index1 += ['Band1']*len(pix_wa_b1)
	type1 += np.ones((1,len(pix_wa_b1))).tolist()[0]
	#land
	pix_l_b1 = arr1[water_mask==2]
	val1 += pix_l_b1.tolist()
	index1 += ['Band1']*len(pix_l_b1)
	type1 += np.zeros((1,len(pix_l_b1))).tolist()[0]
	pix_land_b1 = arr1[np.logical_and(cloud_mask==1,np.logical_and(water_mask!=2,water_mask!=1))]
	val += pix_land_b1.tolist()
	index += ['Band1']*len(pix_land_b1)
	#band2
	#water
	pix_wa_b2 = arr2[water_mask==1]
	val1 += pix_wa_b2.tolist()
	index1 += ['Band2']*len(pix_wa_b2)
	type1 += np.ones((1,len(pix_wa_b2))).tolist()[0]
	#land
	pix_l_b2 = arr2[water_mask==2]
	val1 += pix_l_b2.tolist()
	index1 += ['Band2']*len(pix_l_b2)
	type1 += np.zeros((1,len(pix_l_b2))).tolist()[0]
	pix_land_b2 = arr2[np.logical_and(cloud_mask==1,np.logical_and(water_mask!=2,water_mask!=1))]
	val += pix_land_b2.tolist()
	index += ['Band2']*len(pix_land_b2)
	#band2 - band1
	#water
	pix_wa_min = min1[water_mask==1]
	val1 += pix_wa_min.tolist()
	index1 += ['Band2 - Band1']*len(pix_wa_min)
	type1 += np.ones((1,len(pix_wa_min))).tolist()[0]
	#land
	pix_l_min = min1[water_mask==2]
	val1 += pix_l_min.tolist()
	index1 += ['Band2 - Band1']*len(pix_l_min)
	type1 += np.zeros((1,len(pix_l_min))).tolist()[0]
	pix_land_min = min1[np.logical_and(cloud_mask==1,np.logical_and(water_mask!=2,water_mask!=1))]
	val += pix_land_min.tolist()
	index += ['Band2 - Band1']*len(pix_land_min)
	#b2/b1
	#water
	pix_wa_rat = rat1[water_mask==1]
	val1 += pix_wa_rat.tolist()
	index1 += ['Band2 / Band1']*len(pix_wa_rat)
	type1 += np.ones((1,len(pix_wa_rat))).tolist()[0]
	#land
	pix_l_rat = rat1[water_mask==2]
	val1 += pix_l_rat.tolist()
	index1 += ['Band2 / Band1']*len(pix_l_rat)
	type1 += np.zeros((1,len(pix_l_rat))).tolist()[0]
	pix_land_rat = rat1[np.logical_and(cloud_mask==1,np.logical_and(water_mask!=2,water_mask!=1))]
	val += pix_land_rat.tolist()
	index += ['Band2 / Band1']*len(pix_land_rat)
	#ndvi
	#water
	pix_wa_ndvi = ndvi[water_mask==1]
	val1 += pix_wa_ndvi.tolist()
	index1 += ['ndvi']*len(pix_wa_ndvi)
	type1 += np.ones((1,len(pix_wa_ndvi))).tolist()[0]
	#land
	pix_l_ndvi = ndvi[water_mask==2]
	val1 += pix_l_ndvi.tolist()
	index1 += ['ndvi']*len(pix_l_ndvi)
	type1 += np.zeros((1,len(pix_l_ndvi))).tolist()[0]
	pix_land_ndvi = ndvi[np.logical_and(cloud_mask==1,np.logical_and(water_mask!=2,water_mask!=1))]
	val += pix_land_ndvi.tolist()
	index += ['ndvi']*len(pix_land_ndvi)
	temp['Val'] = val
	temp['Index'] = index
	temp1['Val'] = val1
	temp1['Index'] = index1
	temp1['Types'] = type1
	#random forest train
	df_mod_train = pd.DataFrame()
	for idx in temp1['Index'].unique().tolist():
		df_mod_train[idx] = temp1.loc[temp1['Index']==idx]['Val'].reset_index(drop=True)
	df_mod_train['Types'] = temp1.loc[temp1['Index']==idx]['Types'].reset_index(drop=True)
	df_mod_train = df_mod_train.dropna()
	y = df_mod_train['Types']
	features = list(df_mod_train.columns[:5])
	X = df_mod_train[features]
	class_weight = {1:0.8,0:0.2}
	dt = RandomForestClassifier(n_estimators=50,class_weight=class_weight,n_jobs=-1)
	dt.fit(X,y)
	df_mod_test = pd.DataFrame()
	for idx in temp['Index'].unique().tolist():
		df_mod_test[idx] = temp.loc[temp['Index']==idx]['Val'].reset_index(drop=True)
	df_mod_test = df_mod_test.dropna()
	#random forest predict
	X2 = df_mod_test[features]
	y2_test = dt.predict(X2)
	ds1 = gdal.Open(header+'_b01.tif')
	arr1 = ds1.ReadAsArray()
	geom = ds1.GetGeoTransform()
	arr2 = gdal.Open(header+'_b02.tif').ReadAsArray()
	res_arr = np.ones(arr1.shape)*-1
	try:
		res_arr[np.logical_and(cloud_mask==1,np.logical_and(water_mask!=2,water_mask!=1))] = y2_test
		res_arr[water_mask==1] = 1
		res_arr[water_mask==2] = 0
	except:
		print header
		return
#		res_arr[np.logical_and(miss_dem>215,res_arr==1)]=0
#		res_arr[np.logical_and(np.logical_and(cloud_mask==1,water_mask1==1),res_arr==0)]=1
	driver = gdal.GetDriverByName('GTiff')
	#owl
	dataset = driver.Create(
			'/ssd-scratch/htranvie/Flood/data/results/'+pre+'_'+t+'_bin_new.tif',
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

def Landsat8Training(collected_sample, landsat_image):
	collected_sample_arr = gdal.Open(collected_sample).ReadAsArray()
	temp = pd.DataFrame()
	val = []
	index = []
	types = []
	temp1 = pd.DataFrame()
	tm_files = sorted(glob(landsat_image+'*.[Tt][Ii][Ff]'))
	ds1_tm = gdal.Open(tm_files[0])
	geom_tm = ds1_tm.GetGeoTransform()
	arr_red = ds1_tm.ReadAsArray().astype(np.float)
	arr_nir = gdal.Open(tm_files[1]).ReadAsArray().astype(np.float)
	arr_swir = gdal.Open(tm_files[2]).ReadAsArray().astype(np.float)
	legal_arr_tm = np.logical_and(np.logical_and(arr_red>1,arr_nir>1),#arr_red<11800)
									np.logical_and(arr_red<10000,arr_swir>1))
	#b2-b1
	min_tm1 = np.ones(arr_red.shape)*-9999
	min_tm1[legal_arr_tm] = arr_nir[legal_arr_tm] - arr_red[legal_arr_tm]
	#b2/b1
	rat_tm1 = np.ones(arr_red.shape)*-9999
	rat_tm1[legal_arr_tm] = arr_nir[legal_arr_tm]/(arr_red[legal_arr_tm]).astype(np.float)
	#ndvi
	ndvi_tm = np.ones(arr_red.shape)*-9999
	sum_tm1 = np.ones(arr_red.shape)*-9999
	sum_tm1[legal_arr_tm] = arr_nir[legal_arr_tm] + arr_red[legal_arr_tm]
	ndvi_tm[legal_arr_tm] = min_tm1[legal_arr_tm]/sum_tm1[legal_arr_tm].astype(np.float)
	#ndwi
	ndwi_tm = np.ones(arr_red.shape)*-9999
	rat_tm2 = np.ones(arr_red.shape)*-9999
	rat_tm2[legal_arr_tm] = arr_swir[legal_arr_tm] / arr_nir[legal_arr_tm]
	ndwi_tm[legal_arr_tm] = (np.ones(arr_red.shape)[legal_arr_tm]-rat_tm2[legal_arr_tm])/(np.ones(arr_red.shape)[legal_arr_tm]+rat_tm2[legal_arr_tm])
	#classify
	#band1
	#water
	pix_wa_b1 = arr_red[np.logical_and(legal_arr_tm,collected_sample_arr==1)]
	val += pix_wa_b1.tolist()
	index += ['Band1']*len(pix_wa_b1)
	types += ['Water']*len(pix_wa_b1)
	#land
	pix_land_b1 = arr_red[np.logical_and(legal_arr_tm,collected_sample_arr==0)]
	val += pix_land_b1.tolist()
	index += ['Band1']*len(pix_land_b1)
	types += ['Land']*len(pix_land_b1)
	#band2
	#water
	pix_wa_b2 = arr_nir[np.logical_and(legal_arr_tm,collected_sample_arr==1)]
	val += pix_wa_b2.tolist()
	index += ['Band2']*len(pix_wa_b2)
	types += ['Water']*len(pix_wa_b2)
	#land
	pix_land_b2 = arr_nir[np.logical_and(legal_arr_tm,collected_sample_arr==0)]
	val += pix_land_b2.tolist()
	index += ['Band2']*len(pix_land_b2)
	types += ['Land']*len(pix_land_b2)
	#band2 - band1
	#water
	pix_wa_min = min_tm1[np.logical_and(legal_arr_tm,collected_sample_arr==1)]
	val += pix_wa_min.tolist()
	index += ['Band2 - Band1']*len(pix_wa_min)
	types += ['Water']*len(pix_wa_min)
	#land
	pix_land_min = min_tm1[np.logical_and(legal_arr_tm,collected_sample_arr==0)]
	val += pix_land_min.tolist()
	index += ['Band2 - Band1']*len(pix_land_min)
	types += ['Land']*len(pix_land_min)
	#b2/b1
	#water
	pix_wa_rat = rat_tm1[np.logical_and(legal_arr_tm,collected_sample_arr==1)]
	val += pix_wa_rat.tolist()
	index += ['Band2 / Band1']*len(pix_wa_rat)
	types += ['Water']*len(pix_wa_rat)
	#land
	pix_land_rat = rat_tm1[np.logical_and(legal_arr_tm,collected_sample_arr==0)]
	val += pix_land_rat.tolist()
	index += ['Band2 / Band1']*len(pix_land_rat)
	types += ['Land']*len(pix_land_rat)
	#ndvi
	#water
	pix_wa_ndvi = ndvi_tm[np.logical_and(legal_arr_tm,collected_sample_arr==1)]
	val += pix_wa_ndvi.tolist()
	index += ['ndvi']*len(pix_wa_ndvi)
	types += ['Water']*len(pix_wa_ndvi)
	#land
	pix_land_ndvi = ndvi_tm[np.logical_and(legal_arr_tm,collected_sample_arr==0)]
	val += pix_land_ndvi.tolist()
	index += ['ndvi']*len(pix_land_ndvi)
	types += ['Land']*len(pix_land_ndvi)
	#ndwi
	#water
	pix_wa_ndwi = ndwi_tm[np.logical_and(legal_arr_tm,collected_sample_arr==1)]
	val += pix_wa_ndwi.tolist()
	index += ['ndwi']*len(pix_wa_ndwi)
	types += ['Water']*len(pix_wa_ndwi)
	#land
	pix_land_ndwi = ndwi_tm[np.logical_and(legal_arr_tm,collected_sample_arr==0)]
	val += pix_land_ndwi.tolist()
	index += ['ndwi']*len(pix_land_ndwi)
	types += ['Land']*len(pix_land_ndwi)
	temp['Val'] = val
	temp['Index'] = index
	temp['Types'] = types
	df_mod = temp.copy()
	df_mod1 = pd.DataFrame()
	for idx in df_mod['Index'].unique().tolist():
		df_mod1[idx] = df_mod.loc[df_mod['Index']==idx]['Val'].reset_index(drop=True)
	df_mod1['Types'] = df_mod.loc[df_mod['Index']==idx]['Types'].reset_index(drop=True)
	df_mod1 = df_mod1.dropna()
	types = df_mod1['Types'].tolist()
	int_types = [1 if x=='Water' else 0 for x in types]
	df_mod1['Targets'] = int_types
	y = df_mod1['Targets']
	features = list(df_mod1.columns[:6])
	X = df_mod1[features]
	class_weight = {1:0.7,0:0.3}
	dt_tm = RandomForestClassifier(min_samples_split=10,class_weight=class_weight,random_state=99,n_jobs=-1)
	dt_tm.fit(X,y)
	"""
	classifiers = [
		KNeighborsClassifier(3,n_jobs=-1),
		SVC(kernel="linear", C=0.025),
		SVC(gamma=2, C=1),
		#GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True,n_jobs=-1),
		RandomForestClassifier(min_samples_split=10,class_weight=class_weight,random_state=99,n_jobs=-1),
		MLPClassifier(alpha=0.1),
		GaussianNB()]
	#RandomForestClassifier
	dt_tm_list = []
	for jk,clf in enumerate(classifiers):
		clf.fit(X,y)
		dt_tm_list.append(clf)
		joblib.dump(dt_tm,'/ssd-scratch/htranvie/Flood/data/rf_tm'+str(jk)+'.joblib.pkl',compress=9)
	"""
	return dt_tm

def Landsat8Classify(date0,dt_tm):
	landsat_images = sorted(glob('/ssd-scratch/htranvie/Flood/data/downloads/'+date0.strftime('%Y-%m-%d')+'/LC08_L1TP_*.*'))
	temp = pd.DataFrame()
	val = []
	index = []
	temp1 = pd.DataFrame()
	ds1_tm = gdal.Open(landsat_images[0])
	geom_tm = ds1_tm.GetGeoTransform()
	arr_red = ds1_tm.ReadAsArray().astype(np.float)
	arr_nir = gdal.Open(landsat_images[1]).ReadAsArray().astype(np.float)
	arr_swir = gdal.Open(landsat_images[2]).ReadAsArray().astype(np.float)
	legal_arr_tm = np.logical_and(np.logical_and(arr_red>1,arr_nir>1),#arr_red<11800)
									np.logical_and(arr_red<18000,arr_swir>1))
	#b2-b1
	min_tm1 = np.ones(arr_red.shape)*-9999
	min_tm1[legal_arr_tm] = arr_nir[legal_arr_tm] - arr_red[legal_arr_tm]
	#b2/b1
	rat_tm1 = np.ones(arr_red.shape)*-9999
	rat_tm1[legal_arr_tm] = arr_nir[legal_arr_tm]/(arr_red[legal_arr_tm]).astype(np.float)
	#ndvi
	ndvi_tm = np.ones(arr_red.shape)*-9999
	sum_tm1 = np.ones(arr_red.shape)*-9999
	sum_tm1[legal_arr_tm] = arr_nir[legal_arr_tm] + arr_red[legal_arr_tm]
	ndvi_tm[legal_arr_tm] = min_tm1[legal_arr_tm]/sum_tm1[legal_arr_tm].astype(np.float)
	#ndwi
	ndwi_tm = np.ones(arr_red.shape)*-9999
	rat_tm2 = np.ones(arr_red.shape)*-9999
	rat_tm2[legal_arr_tm] = arr_swir[legal_arr_tm] / arr_nir[legal_arr_tm]
	ndwi_tm[legal_arr_tm] = (np.ones(arr_red.shape)[legal_arr_tm]-rat_tm2[legal_arr_tm])/(np.ones(arr_red.shape)[legal_arr_tm]+rat_tm2[legal_arr_tm])
	#classify
	#band1
	val += arr_red[legal_arr_tm].tolist()
	index += ['Band1']*np.sum(legal_arr_tm)
	#band2
	val += arr_nir[legal_arr_tm].tolist()
	index += ['Band2']*np.sum(legal_arr_tm)
	#band2 - band1
	val += min_tm1[legal_arr_tm].tolist()
	index += ['Band2 - Band1']*np.sum(legal_arr_tm)
	#b2/b1
	val += rat_tm1[legal_arr_tm].tolist()
	index += ['Band2 / Band1']*np.sum(legal_arr_tm)
	#ndvi
	val += ndvi_tm[legal_arr_tm].tolist()
	index += ['ndvi']*np.sum(legal_arr_tm)
	#ndwi
	val += ndwi_tm[legal_arr_tm].tolist()
	index += ['ndwi']*np.sum(legal_arr_tm)
	temp['Val'] = val
	temp['Index'] = index
	df_mod0 = temp.copy()
	df_mod01 = pd.DataFrame()
	for idx in df_mod0['Index'].unique().tolist():
		df_mod01[idx] = df_mod0.loc[df_mod0['Index']==idx]['Val'].reset_index(drop=True)
	df_mod01 = df_mod01.dropna()
	features1 = list(df_mod01.columns[:6])
	X_test = df_mod01[features1]
	y_test = dt_tm.predict(X_test)
	res_arr_tm = np.ones(arr_red.shape)*-99
	res_arr_tm[legal_arr_tm] = y_test
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(
			'/ssd-scratch/htranvie/Flood/data/downloads/'+date0.strftime('%Y-%m-%d')+'/Landsat_'+date0.strftime('%Y%m%d')+'_bin.tif',
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

def ResampleLandsat(img,img1):
	os.system('gdalwarp -overwrite -r cubicspline -srcnodata -99 -dstnodata -99 -tr 0.00219684 0.00219684 '+img+' '+img1)

def LandsatValidation(d0,createTif=None):
	ds_mod = gdal.Open('/ssd-scratch/htranvie/Flood/data/results/MOD_'+d0.strftime('%Y%m%d')+'_bin_new.tif')
	arr_mod = ds_mod.ReadAsArray()
	arr_mod_flood = gdal.Open(glob('/ssd-scratch/htranvie/Flood/data/downloads/'+d0.strftime('%Y-%m-%d')+'/MWP_*.tif')[0]).ReadAsArray()
	arr_tm = gdal.Open(glob('/ssd-scratch/htranvie/Flood/data/downloads/'+d0.strftime('%Y-%m-%d')+'/Landsat_*_resampled.tif')[0]).ReadAsArray()
	results = pd.DataFrame()
	val = []
	type = []
	im_type = []
	date = []
	nan_arr_mod = np.logical_or(np.logical_or(arr_mod==-1,arr_mod_flood==0),arr_tm==-99).astype(np.int)
	#hit
	hit_arr = np.logical_and(arr_mod[nan_arr_mod!=1]==1,arr_tm[nan_arr_mod!=1]==1)
	no_hit = np.sum(hit_arr)
	#miss
	miss_arr = np.logical_and(arr_mod[nan_arr_mod!=1]==0,arr_tm[nan_arr_mod!=1]==1)
	no_miss = np.sum(miss_arr)
	#false
	false_arr = np.logical_and(arr_mod[nan_arr_mod!=1]==1,arr_tm[nan_arr_mod!=1]==0)
	no_false = np.sum(false_arr)
	#correct negative
	corrneg = np.logical_and(arr_mod[nan_arr_mod!=1]==0,arr_tm[nan_arr_mod!=1]==0)
	no_corrneg = np.sum(corrneg)
	val += [no_hit,no_miss,no_false,no_corrneg]
	type += ['Hit','Miss','False','Correct Negative']
	val += [no_hit/(float(no_miss)+no_hit), no_false/(float(no_hit)+no_false), (no_hit*no_corrneg-no_false*no_miss)/float((no_hit+no_miss)*(no_corrneg+no_false))]
	type += ['POD','FAR','HK']
	im_type += ['MOD']*7
	#for mod flood images
	#hit
	hit_arr1 = np.logical_and(arr_mod_flood[nan_arr_mod!=1]>1,arr_tm[nan_arr_mod!=1]==1)
	no_hit1 = np.sum(hit_arr1)
	#miss
	miss_arr1 = np.logical_and(arr_mod_flood[nan_arr_mod!=1]==1,arr_tm[nan_arr_mod!=1]==1)
	no_miss1 = np.sum(miss_arr1)
	#false
	false_arr1 = np.logical_and(arr_mod_flood[nan_arr_mod!=1]>1,arr_tm[nan_arr_mod!=1]==0)
	no_false1 = np.sum(false_arr1)
	#correct negative
	corrneg1 = np.logical_and(arr_mod_flood[nan_arr_mod!=1]==1,arr_tm[nan_arr_mod!=1]==0)
	no_corrneg1 = np.sum(corrneg1)
	val += [no_hit1,no_miss1,no_false1,no_corrneg1]
	type += ['Hit','Miss','False','Correct Negative']
	val += [no_hit1/(float(no_miss1)+no_hit1), no_false1/(float(no_hit1)+no_false1), (no_hit1*no_corrneg1-no_false1*no_miss1)/float((no_hit1+no_miss1)*(no_corrneg1+no_false1))]
	type += ['POD','FAR','HK']
	im_type += ['MWP']*7
	date += [d0.strftime('%Y-%m-%d')]*14
	results['Date'] = date
	results['Im_type'] = im_type
	results['Types'] = type
	results['Values'] = val
	if createTif=='OK':
		res_arr = np.zeros(arr_mod.shape)
		res_arr[nan_arr_mod!=1] = hit_arr+miss_arr*2+false_arr*3
		res_arr1 = np.zeros(arr_mod.shape)
		res_arr1[nan_arr_mod!=1] = hit_arr1+miss_arr1*2+false_arr1*3
		driver = gdal.GetDriverByName('GTiff')
		dataset1 = driver.Create(
				'/ssd-scratch/htranvie/Flood/data/downloads/'+d0.strftime('%Y-%m-%d')+'/MOD_'+d0.strftime('%Y%m%d')+'_cat.tif',
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
		dataset2 = driver.Create(
				'/ssd-scratch/htranvie/Flood/data/downloads/'+d0.strftime('%Y-%m-%d')+'/MWP_'+d0.strftime('%Y%m%d')+'_cat.tif',
				res_arr1.shape[1],
				res_arr1.shape[0],
				1,
				gdal.GDT_Int16,
				)
		dataset2.SetGeoTransform(ds_mod.GetGeoTransform())
		outband2 = dataset2.GetRasterBand(1)
		outband2.WriteArray(res_arr1)
		outband2.FlushCache()
		outband2.SetNoDataValue(0)
		dataset2.FlushCache()
	return results
	
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
dt = joblib.load('/ssd-scratch/htranvie/Flood/data/rf.joblib.pkl')
d0 = date(2014,6,17)

for i in [0,2,18,23,50,57,59]:
	for pre in ['MOD']:
		ClassifyMODIS(d0,i,pre)

for i in [0,2,18,23,50,57,59]:
	d1 = d0+timedelta(days=i)
	ReclassifyMODIS(d1,'MOD')

for i in [0,2,18,23,50,59]:
	d1 = d0+timedelta(days=i)
	tm_path = '/ssd-scratch/htranvie/Flood/data/downloads/'+d1.strftime("%Y-%m-%d")+'/'
	dt_tm = Landsat8Training(tm_path+'collected_samples_'+d1.strftime("%Y%m%d")+'.tif',
			tm_path+'LC08_L1TP_*'+d1.strftime("%Y%m%d"))
	Landsat8Classify(d1,dt_tm)

tot_res = pd.DataFrame()
for i in [0,2,18,23,50,57,59]:
	d1 = d0+timedelta(days=i)
	file_path = '/ssd-scratch/htranvie/Flood/data/downloads/'+d1.strftime('%Y-%m-%d')+'/'
	tm_file = glob(file_path+'Landsat_*_bin.tif')[0]
	tm_resampled_file = tm_file.split('.')[0]+'_resampled.tif'
	ResampleLandsat(tm_file,tm_resampled_file)
	pd_res = LandsatValidation(d1,'OK')
	tot_res = tot_res.append(pd_res, ignore_index=True)


















