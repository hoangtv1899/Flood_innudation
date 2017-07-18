#!/data/apps/anaconda/2.7-4.3.1/bin/python

import scipy.io
import os
import numpy as np
from ClearCloud import ClearCloud
from glob import glob
from datetime import datetime, timedelta, date
import gdal
import scipy.ndimage as ndimage
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap

list_MOD = glob('/ssd-scratch/htranvie/Flood/data/results/M*D_'+str(yr)+'*_bin.tif')
arr_temp = gdal.Open(list_MOD[0]).ReadAsArray()
d0 = datetime(2008,6,5)
oyflood = np.array([]).reshape(0,arr_temp.shape[0],arr_temp.shape[1])
for i in range(15):
	d = (d0+timedelta(days=i)).strftime('%Y%m%d')
	bin_files = sorted([x for x in list_MOD if d in x])
	if not bin_files:
		oyflood = np.concatenate([oyflood,np.zeros((2,arr_temp.shape[0],arr_temp.shape[1]))],axis=0)
	elif len(bin_files) == 1:
		if 'MOD' in bin_files[0]:
			arr0 = gdal.Open(bin_files[0]).ReadAsArray()
			oyflood = np.concatenate([oyflood,arr0.reshape(-1,arr_temp.shape[0],arr_temp.shape[1])],axis=0)
			oyflood = np.concatenate([oyflood,np.zeros((1,arr_temp.shape[0],arr_temp.shape[1]))],axis=0)
		else:
			arr0 = gdal.Open(bin_files[0]).ReadAsArray()
			oyflood = np.concatenate([oyflood,np.zeros((1,arr_temp.shape[0],arr_temp.shape[1]))],axis=0)
			oyflood = np.concatenate([oyflood,arr0.reshape(-1,arr_temp.shape[0],arr_temp.shape[1])],axis=0)
	else:
		for bin_file in bin_files:
			arr1 = gdal.Open(bin_file).ReadAsArray()
			oyflood = np.concatenate([oyflood,arr1.reshape(-1,arr_temp.shape[0],arr_temp.shape[1])],axis=0)

oyflood[oyflood==1]=2
oyflood[oyflood==-1]=1
scipy.io.savemat('/ssd-scratch/htranvie/Flood/data/oyflood',mdict={'oyflood':oyflood})

oyscacld = scipy.io.loadmat('/ssd-scratch/htranvie/Flood/data/oyflood.mat')['oyflood']
oyscacld1 = ndimage.zoom(oyscacld,(1,4,4),order=0)
sca = ClearCloud(oyscacld1,date(2008,6,5),date(2008,6,5)+timedelta(days=29))
scipy.io.savemat('/ssd-scratch/htranvie/Flood/data/sca_py3',mdict={'sca_py3':sca})

sca = scipy.io.loadmat('/ssd-scratch/htranvie/Flood/data/sca_py2.mat')['sca_py2']
#for matlab
sca = scipy.io.loadmat('/ssd-scratch/htranvie/Flood/data/sca2.mat')['sca2']
sca = np.swapaxes(np.swapaxes(sca,0,2),1,2)
#Draw report
t0 = date(2008,6,5)
fig=plt.figure()
m1 = 4
n = 6
for i in range(n*m1,n*m1+n):
	t = t0+timedelta(days=i/2)
	
	ax=fig.add_subplot(2,n,i-n*m1+1)
	m = Basemap(llcrnrlon=-91.3,llcrnrlat=40.60,urcrnrlon=-90.75,urcrnrlat=41.2,\
				projection='mill',lon_0=0)
	m.readshapefile('/ssd-scratch/htranvie/Flood/shapes/SWBD_mississippi2',
							'SWBD_mississippi',linewidth=0.8/float(n), color='k',zorder=1)
	if np.unique(oyscacld[i,:,:]).shape == (3,):
		cmap1 = LinearSegmentedColormap.from_list('mycmap', [(0 / 2., (0, 0, 0, 0)),
															(1 / 2., 'grey'),
															(2 / 2., 'blue')]
														)
	elif np.unique(oyscacld[i,:,:]).shape == (2,):
		cmap1 = LinearSegmentedColormap.from_list('mycmap', [(0 / 2., (0, 0, 0, 0)),
															(2 / 2., 'grey')]
														)
#	elif np.unique(oyscacld[i,:,:]).shape == (1,):
#		cmap1 = LinearSegmentedColormap.from_list('mycmap', [(2 / 2., 'grey')])
	m.imshow(np.flipud(oyscacld[i,:,:]), cmap=cmap1,zorder=2)
	if i%2==0:
		plt.title(t.strftime('%Y-%m-%d')+'\n AM',fontsize=40/n)
	else:
		plt.title(t.strftime('%Y-%m-%d')+'\n PM',fontsize=40/n)
	ax=fig.add_subplot(2,n,i-n*m1+n+1)
	m = Basemap(llcrnrlon=-91.3,llcrnrlat=40.60,urcrnrlon=-90.75,urcrnrlat=41.2,\
				projection='mill',lon_0=0)
	m.readshapefile('/ssd-scratch/htranvie/Flood/shapes/SWBD_mississippi2',
							'SWBD_mississippi',linewidth=0.8/float(n), color='k',zorder=1)
	cmap = LinearSegmentedColormap.from_list('mycmap', [(0 / 2., (0, 0, 0, 0)),
														(2 / 2., 'blue')]
													)
	m.imshow(np.flipud(sca[i,:,:]), cmap=cmap,zorder=2)

fig.savefig('h'+str(m1+5)+'.png',dpi=800)


























