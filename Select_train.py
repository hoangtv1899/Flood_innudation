#!/data/apps/enthought_python/2.7.3/bin/python

import scipy.io
import os
import numpy as np
from osgeo import gdal, gdalnumeric, ogr
from datetime import date, datetime, timedelta
from glob import glob
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
mpl.use('Agg')
import matplotlib.pyplot as plt

water_mask = gdal.Open('/ssd-scratch/htranvie/Flood/data/elevation/SWBD_mississippi_resampled3.tif').ReadAsArray()
d0 = date(2000,2,24)

w_b1 = np.zeros(water_mask.shape)

for i in range(367):
	t = (d0+timedelta(days=i)).strftime('%Y%m%d')
	for pre in ['MOD']:
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
		
		w_b1 += np.logical_and(cloud_mask==1,water_mask==1)

geom = gdal.Open(header+'_b01.tif').GetGeoTransform()
driver = gdal.GetDriverByName('GTiff')
#owl
dataset = driver.Create(
		'train_res1.tif',
		w_b1.shape[1],
		w_b1.shape[0],
		1,
		gdal.GDT_Int16,
		)
dataset.SetGeoTransform(geom)
outband = dataset.GetRasterBand(1)
outband.WriteArray(w_b1)
outband.FlushCache()
outband.SetNoDataValue(0)
dataset.FlushCache()