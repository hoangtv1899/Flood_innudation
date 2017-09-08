#!/data/apps/anaconda/2.7-4.3.1/bin/python

import scipy.io
import os
import numpy as np
from osgeo import gdal, gdalnumeric, ogr, osr
from datetime import date, datetime, timedelta
from glob import glob
import scipy.ndimage
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
plt.style.use('ggplot')

d0 = date(2013,7,25)
d2 = date(2014,6,17)
for i in [0,2,18,23,50,57,59]:#[0,16,23,32,34]:
	d1 = d2 + timedelta(days=i)
	file_path = '/ssd-scratch/htranvie/Flood/data/downloads/'+d1.strftime('%Y-%m-%d')+'/'
	im_files = sorted(glob(file_path+'*_cat.tif'))
	ds0 = gdal.Open(im_files[0])
	#Get extend
	gt = ds0.GetGeoTransform()
	# get the edge coordinates and add half the resolution 
	# to go to center coordinates
	xres, yres = gt[1],gt[5]
	xmin = gt[0] + xres * 0.5
	xmax = gt[0] + (xres * ds0.RasterXSize) - xres * 0.5
	ymin = gt[3] + (yres * ds0.RasterYSize) + yres * 0.5
	ymax = gt[3] - yres * 0.5
	
	#Create basemap
	fig=plt.figure()
	for j in range(2):
		data = gdal.Open(im_files[j]).ReadAsArray()
		ax=fig.add_subplot(1,2,j+1)
		m = Basemap(llcrnrlon=xmin,llcrnrlat=ymin,urcrnrlon=xmax,urcrnrlat=ymax,\
					projection='mill',lon_0=0)
#		m.readshapefile('/ssd-scratch/htranvie/Flood/shapes/swbd_iowa2',
#							'swbd_iowa2',linewidth=0.2, color='k',zorder=1)
#		x = np.linspace(0, m.urcrnrx, data.shape[1])
#		y = np.linspace(0, m.urcrnry, data.shape[0])
#		xx,yy = np.meshgrid(x,y)
		cmap1 = LinearSegmentedColormap.from_list('mycmap', [(0 / 3., (0, 0, 0, 0)),
																(1/3., 'blue'),
																(2 / 3., 'red'),
																(3 / 3., 'green')]
															)
		m.imshow(data,cmap=cmap1)
	fig.savefig('h_'+d1.strftime('%Y%m%d')+'.png',dpi=400)
	fig.clf()
	

