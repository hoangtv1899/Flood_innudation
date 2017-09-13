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

def train(arr1, arr2, cloud_mask, water_mask,land_type=0, arr3 = None):
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
	temp['Val'] = val1
	temp['Index'] = index1
	temp['Types'] = type1
	df_mod_train = pd.DataFrame()
	for idx in temp['Index'].unique().tolist():
		df_mod_train[idx] = temp.loc[temp['Index']==idx]['Val'].reset_index(drop=True)
	df_mod_train['Types'] = temp.loc[temp['Index']==idx]['Types'].reset_index(drop=True)
	df_mod_train = df_mod_train.dropna()
	return df_mod_train

def test(arr1, arr2, cloud_mask, arr3 = None):
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
	corrneg = np.logical_and(mod_arr[mask]==l,ref_arr[mask]==0)
	no_corrneg = np.sum(corrneg)
	val += [no_hit,no_miss,no_false,no_corrneg]
	type += ['Hit','Miss','False','Correct Negative']
	val += [no_hit/(float(no_miss)+no_hit), no_false/(float(no_hit)+no_false), (no_hit*no_corrneg-no_false*no_miss)/float((no_hit+no_miss)*(no_corrneg+no_false))]
	type += ['POD','FAR','HK']
	im_type += [type_name]*7
	res_arr = np.zeros(mod_arr.shape)
	res_arr[mask] = hit_arr+miss_arr*2+false_arr*3
	return val, type, im_type, res_arr

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

def ClassifyMODIS(d1,pre,rf):
	t = d1.strftime('%Y%m%d')
	header = 'clipped_data/'+pre+'09GQ.A'+t
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
	MODClassTest = test(arr1, arr2, cloud_mask)
	#random forest predict
	features = list(MODClassTest.columns[:5])
	X2 = MODClassTest[features]
	y2_test = rf.predict(X2)
	ds1 = gdal.Open(header+'_b01.tif')
	geom = ds1.GetGeoTransform()
	res_arr = np.ones(arr1.shape)*-1
	try:
		res_arr[cloud_mask==1] = y2_test
	except:
		print header
		return
	createTif('results/'+pre+'_'+t+'_bin.tif', res_arr, geom)

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

def ReclassifyMODIS(d0,pre,random_state=None,dem_arr=None):
	t = d0.strftime('%Y%m%d')
	header = 'clipped_data/'+pre+'09GQ.A'+t
	try:
		arr1 = gdal.Open(header+'_b01.tif').ReadAsArray()
		arr2 = gdal.Open(header+'_b02.tif').ReadAsArray()
		water_mask = gdal.Open('results/'+pre+'_'+t+'_bin.tif').ReadAsArray()
	except:
		return
	n_pix_water = np.sum(water_mask==1)
	if n_pix_water < 100:
		return
	#random select land pixels
	xx,yy = np.where(np.logical_and(water_mask==0,arr1<1700))
	xx1,yy1 = np.where(water_mask==1)
	pix_land = np.random.randint(0,xx.shape[0],n_pix_water).tolist()
	pix_water_discard = np.random.randint(0,xx1.shape[0],3*n_pix_water/4).tolist()
	for pix in pix_land:
		water_mask[xx[pix],yy[pix]] = 2
	for pix1 in pix_water_discard:
		water_mask[xx1[pix1],yy1[pix1]] = 0
	cloud_mask = np.logical_and(arr1+arr2 !=0,
								np.logical_and(np.logical_and(arr2!=-28672,arr1!=-28672),
								np.logical_and(arr1!=0,arr1<1700))).astype(np.int)
	cloud_mask_test = np.logical_and(cloud_mask==1,np.logical_and(water_mask!=2,water_mask!=1))
	if len(np.unique(cloud_mask))==1:
		return
	MODClassTrain = train(arr1, arr2, cloud_mask, water_mask, 2)
	MODClassTest = test(arr1, arr2, cloud_mask_test)
	#random forest train
	y = MODClassTrain['Types']
	features = list(MODClassTrain.columns[:5])
	X = MODClassTrain[features]
	class_weight = {1:0.7,0:0.3}
	dt = RandomForestClassifier(n_estimators=50,class_weight=class_weight,n_jobs=-1,random_state=random_state)
	dt.fit(X,y)
	#random forest predict
	X2 = MODClassTest[features]
	y2_test = dt.predict(X2)
	ds1 = gdal.Open(header+'_b01.tif')
	geom = ds1.GetGeoTransform()
	res_arr = np.ones(arr1.shape)*-1
	try:
		res_arr[cloud_mask_test] = y2_test
		res_arr[water_mask==1] = 1
		res_arr[water_mask==2] = 0
		if dem_arr is not None:
			res_arr[np.logical_and(dem_arr>195,res_arr==1)] = 0
	except:
		print header
		return
	if os.path.isfile('results/'+pre+'_'+t+'_bin_new.tif'):
		os.remove('results/'+pre+'_'+t+'_bin_new.tif')
	createTif('results/'+pre+'_'+t+'_bin_new.tif', res_arr, geom)

def Landsat8Training(collected_sample, landsat_image):
	collected_sample_arr = gdal.Open(collected_sample).ReadAsArray()
	tm_files = sorted(glob(landsat_image+'*.[Tt][Ii][Ff]'))
	arr_red = gdal.Open(tm_files[0]).ReadAsArray().astype(np.float)
	arr_nir = gdal.Open(tm_files[1]).ReadAsArray().astype(np.float)
	arr_swir = gdal.Open(tm_files[2]).ReadAsArray().astype(np.float)
	legal_arr_tm = np.logical_and(np.logical_and(arr_red>1,arr_nir>1),#arr_red<11800)
									np.logical_and(arr_red<10000,arr_swir>1))
	landsatClass = train(arr_red, arr_nir, legal_arr_tm, collected_sample_arr, arr_swir)
	y = landsatClass['Types']
	features = list(landsatClass.columns[:6])
	X = landsatClass[features]
	class_weight = {1:0.7,0:0.3}
	dt_tm = RandomForestClassifier(min_samples_split=10,class_weight=class_weight,random_state=99,n_jobs=-1)
	dt_tm.fit(X,y)
	return dt_tm

def Landsat8Classify(date0,dt_tm):
	landsat_images = sorted(glob('Landsat/LC08_L1TP_'+date0.strftime('%Y%m%d')+'*.[tT][iI][fF]'))
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

def ResampleLandsat(img,img1):
	os.system('gdalwarp -overwrite -r cubicspline -srcnodata -99 -dstnodata -99 -tr 0.00219684 0.00219684 '+img+' '+img1)

def LandsatValidation(d0,createTif=None):
	ds_mod = gdal.Open('/ssd-scratch/htranvie/Flood/data/results/MOD_'+d0.strftime('%Y%m%d')+'_bin_new.tif')
	arr_mod = ds_mod.ReadAsArray()
	arr_mod_flood = gdal.Open(glob('/ssd-scratch/htranvie/Flood/data/downloads/'+d0.strftime('%Y-%m-%d')+'/MWP_*_1D1OS.tif')[0]).ReadAsArray()
	arr_tm = gdal.Open(glob('/ssd-scratch/htranvie/Flood/data/downloads/'+d0.strftime('%Y-%m-%d')+'/Landsat_*_resampled.tif')[0]).ReadAsArray()
	results = pd.DataFrame()
	nan_arr_mod = np.logical_or(np.logical_or(arr_mod==-1,arr_mod_flood==0),arr_tm==-99).astype(np.int)
	val_mod, type_mod, im_mod, res_mod = catValidation(arr_tm, arr_mod, 'MOD',
														np.invert(nan_arr_mod))
	#for mod flood images
	val_modf, type_modf, im_modf, res_modf = catValidation(arr_tm, arr_mod_flood, 'MWP',
														np.invert(nan_arr_mod),0,2)
	date1 += [d0.strftime('%Y-%m-%d')]*14
	results['Date'] = date1
	results['Im_type'] = im_mod+im_modf
	results['Types'] = type_mod+type_modf
	results['Values'] = val_mod+val_modf
	if createTif=='OK':
		createTif('/ssd-scratch/htranvie/Flood/data/downloads/'+d0.strftime('%Y-%m-%d')+'/MOD_'+d0.strftime('%Y%m%d')+'_cat.tif', 
					res_mod, 
					ds_mod.GetGeoTransform(), 
					gdal.GDT_Int16, 
					0)
		createTif('/ssd-scratch/htranvie/Flood/data/downloads/'+d0.strftime('%Y-%m-%d')+'/MWP_'+d0.strftime('%Y%m%d')+'_cat.tif', 
					res_modf, 
					ds_mod.GetGeoTransform(), 
					gdal.GDT_Int16, 
					0)
	return results

def LandsatValidationMODIS(d0,pre,createTif=None):
	ds_mod = gdal.Open('Cloud_free/'+pre+'_FM'+d0.strftime('%Y%m%d')+'.tif')
	arr_mod = ds_mod.ReadAsArray()
	arr_tm = gdal.Open(glob('Landsat/Landsat_*'+d0.strftime('%Y%m%d')+'*_resampled.tif')[0]).ReadAsArray()
	list_org = glob('results/'+pre+'_'+d0.strftime('%Y%m%d')+'*.tif')
	arr_mod_flood_file = [x for x in list_org if 'new' in x][0]
	if not arr_mod_flood_file:
		arr_mod_flood = gdal.Open(list_org[0]).ReadAsArray()
	else:
		arr_mod_flood = gdal.Open(arr_mod_flood_file).ReadAsArray()
	results = pd.DataFrame()
	nan_arr_mod = (arr_tm!=-99).astype(np.int)
	val_mod, type_mod, im_mod, res_mod = catValidation(arr_tm, arr_mod, 'Cloud_free_MOD',
														nan_arr_mod)
	#for images with cloud
	val_modf, type_modf, im_modf, res_modf = catValidation(arr_tm, arr_mod_flood, 'MOD',
														nan_arr_mod,1,1)
	date1 += [d0.strftime('%Y-%m-%d')]*14
	results['Date'] = date1
	results['Im_type'] = im_mod + im_modf
	results['Types'] = type_mod + type_modf
	results['Values'] = val_mod + val_modf
	if createTif=='OK':
		createTif('validation/Cloud_free_'+pre+'_'+d0.strftime('%Y%m%d')+'_cat.tif', 
					res_mod, 
					ds_mod.GetGeoTransform(), 
					gdal.GDT_Int16, 
					0)
		createTif('validation/'+pre+'_'+d0.strftime('%Y%m%d')+'_cat.tif', 
					res_modf, 
					ds_mod.GetGeoTransform(), 
					gdal.GDT_Int16, 
					0)
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

rf = joblib.load('rf9.joblib.pkl')
d0 = date(2013,6,7)

for i in [0,48,64,416]:
	d1 = d0+timedelta(days=i)
	tm_path = 'Landsat/'
	dt_tm = Landsat8Training(tm_path+'collected_samples_'+d1.strftime("%Y%m%d")+'.tif',
			tm_path+'LC08_L1TP_*'+d1.strftime("%Y%m%d"))
	Landsat8Classify(d1,dt_tm)

d0 = date(2017,8,18)
dem_arr = gdal.Open('elevation/iowa_dem_resampled.tif').ReadAsArray()
tot_res = pd.DataFrame()
for i in range(21):
	d1 = d0+timedelta(days=i)
	for pre in ['MOD','MYD']:
		ClassifyMODIS(d1,pre,rf)
		ReclassifyMODIS(d1,pre,89)
#		file_path = '/ssd-scratch/htranvie/Flood/data/downloads/'+d1.strftime('%Y-%m-%d')+'/'
#		tm_file = glob(file_path+'Landsat_*_bin.tif')[0]
#		tm_resampled_file = tm_file.split('.')[0]+'_resampled.tif'
#		ResampleLandsat(tm_file,tm_resampled_file)
		pd_res = LandsatValidationMODIS(d1,'MOD','OK')
		tot_res = tot_res.append(pd_res, ignore_index=True)
		tot_res.to_csv('data_2013.csv',index=False)

d2 = date(2014,6,17)
tot_res1 = pd.DataFrame()
for i in [0,2,18,23,50,57,59]:
	d3 = d2+timedelta(days=i)
	for pre in ['MOD']:
#		ClassifyMODIS(d2,i,pre)
		ReclassifyMODIS(d3,pre)
#		file_path = '/ssd-scratch/htranvie/Flood/data/downloads/'+d1.strftime('%Y-%m-%d')+'/'
#		tm_file = glob(file_path+'Landsat_*_bin.tif')[0]
#		tm_resampled_file = tm_file.split('.')[0]+'_resampled.tif'
#		ResampleLandsat(tm_file,tm_resampled_file)
		pd_res = LandsatValidation(d3,'OK')
#			pod_mod,pod_mwp = pd_res[pd_res.Types=='POD'].Values.tolist()
		tot_res1 = tot_res1.append(pd_res, ignore_index=True)
#			if pod_mod > pod_mwp:
#				count += 1
#	print count
		
		tot_res1.to_csv('data_2014.csv',index=False)



#Write cloud free modis
t_mid = date(2014,7,28)
#refl_vi = scipy.io.loadmat('refl_vi'+t_mid.strftime('%Y%m%d')+'.mat')['refl_vi'+t_mid.strftime('%Y%m%d')]
#WriteMODIS(date(2013,8,2),t_mid,refl_vi)
LandsatBin = glob('Landsat/Landsat_'+t_mid.strftime('%Y%m%d')+'_bin.tif')[0]
LandsatResampled = LandsatBin.split('.')[0]+'_resampled.tif'
#ResampleLandsat(LandsatBin, LandsatResampled)
pd_res = LandsatValidationMODIS(t_mid,'MOD','OK')
pd_res1 = LandsatValidationMODIS(t_mid,'MYD','OK')
pd_res
pd_res1
