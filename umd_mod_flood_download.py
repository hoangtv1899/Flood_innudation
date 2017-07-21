#!/data/apps/enthought_python/2.7.3/bin/python

import os
import numpy as np
from StringIO import StringIO
import gzip
from datetime import datetime, date, timedelta as td
import urllib2
from glob import glob
import gdal

ftp_path = 'ftp://ftp.glcf.umd.edu/glcf/MODIS_Flood_Maps/ft/ft0001/GLCF.TSM.DF-001.00/midwest08/'

request = urllib2.Request(ftp_path)
response = urllib2.urlopen(request)
list_dirs = response.read().splitlines()
list_dirs = [x.split(' ')[-1] for x in list_dirs if 'met' not in x]

for mod_dir in list_dirs:
	tif_file = mod_dir+'.Geog.tif'
	f = urllib2.urlopen(ftp_path+mod_dir+'/'+tif_file+'.gz')
	comp_file = StringIO()
	comp_file.write(f.read())
	comp_file.seek(0)
	zipfile = gzip.GzipFile(fileobj=comp_file,mode='rb')
	with open('/share/ssd-scratch/htranvie/Flood/data/mod_flood/'+tif_file, 'w') as code:
		code.write(zipfile.read())

mod_flood_files = glob('/share/ssd-scratch/htranvie/Flood/data/mod_flood/*.tif')
geom = gdal.Open(mod_flood_files[0]).GetGeoTransform()
geom = list(geom)
geom[3] = geom[3]-.148
geom[0] = geom[0]-0.021

for mod_flood_file in mod_flood_files:
	dt = os.path.basename(mod_flood_file).split('.')[1]
	tif_file1 = '/ssd-scratch/htranvie/Flood/data/mod_flood/geotiff/MOD_FLOOD.'+datetime.strptime(dt,'%Y%j').strftime('%Y%m%d')+'.tif'
	clipped_tif_file1 = '/ssd-scratch/htranvie/Flood/data/mod_flood/clipped_data/MOD_FLOOD.'+datetime.strptime(dt,'%Y%j').strftime('%Y%m%d')+'.tif'
	arr = gdal.Open(mod_flood_file).ReadAsArray()
	temp_arr = np.zeros((arr.shape[1],arr.shape[2]))
	temp_arr[np.logical_and(np.logical_and(arr[0,:,:]==255,arr[1,:,:]==255),arr[2,:,:]==255)] = 1
	temp_arr[np.logical_or(np.logical_and(arr[1,:,:]==0,arr[2,:,:]==255),
							np.logical_and(arr[1,:,:]==0,arr[0,:,:]==255))] = 2
	driver = gdal.GetDriverByName('GTiff')
	#owl
	dataset = driver.Create(
			tif_file1,
			temp_arr.shape[1],
			temp_arr.shape[0],
			1,
			gdal.GDT_Int16,
			)
	dataset.SetGeoTransform(geom)
	outband = dataset.GetRasterBand(1)
	outband.WriteArray(temp_arr)
	outband.FlushCache()
	outband.SetNoDataValue(-99)
	dataset.FlushCache()
	os.system('gdal_translate -r cubic -projwin -91.2999999999999687 41.1999999999999886 -90.7512158899999690 40.5991081599999859 -tr 0.00270337 0.00208643 '+tif_file1+' '+clipped_tif_file1)
	