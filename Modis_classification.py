#!/data/apps/anaconda/2.7-4.3.1/bin/python

import scipy.io
import os
import numpy as np
from osgeo import gdal, gdalnumeric, ogr, osr
from datetime import date, datetime, timedelta
from glob import glob
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

#set threshold for water
t0 = date(2013,7,18)
#t0 = date(2014,7,21)
list_days = [(t0+timedelta(days=i)).strftime('%Y%m%d') for i in range(41)]
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
C = 1
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
		createTif('results_test/'+pre+'_'+d+'_bindem.tif',res_arr,geom)
		





