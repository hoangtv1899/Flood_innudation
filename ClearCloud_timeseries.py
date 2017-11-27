#!/data/apps/anaconda/2.7-4.3.1/bin/python

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
import os
import numpy as np
from datetime import date, datetime, timedelta
from ClearCloud import ClearCloud
from glob import glob
import scipy.ndimage

d0 = date(2013,6,14)
for i in [16,32,48,57,73]:
#for i in [377,409,425,441]:
	d1 = d0 + timedelta(days=i)
#	if os.path.isfile('Cloud_free/MOD_CF_'+d1.strftime('%Y%m%d')+'.npy'):
#		continue
	selectedDays = [(d1+timedelta(days=-x-1)).strftime('%Y%m%d') for x in range(7)][::-1]+\
					[(d1+timedelta(days=x)).strftime('%Y%m%d') for x in range(8)]
	refl_in = np.array([]).reshape(0,1494,1836)
	count = 0
	for j,day in enumerate(selectedDays):
		if os.path.isfile('results/MOD_'+day+'_bin.tif'):
			arr = io.imread('results/MOD_'+day+'_bin.tif')
		else:
			count += 1
			arr = np.ones((1494,1836))*-1
		refl_in = np.vstack([arr[np.newaxis,:,:],refl_in])
	if count > 7:
		continue
	refl_in[refl_in==1] = 2
	refl_in[refl_in==-1] = 1
	#L = scipy.ndimage.zoom(refl_in,(1,2,2),order=1)
	
	refl_vi = ClearCloud(refl_in, d1+timedelta(days=-7), d1+timedelta(days=7))
	plt.clf()
	plt.imshow(np.ma.masked_where(refl_vi==-99,refl_vi))
	plt.savefig('Cloud_free/img/h'+d1.strftime('%Y%m%d')+'.png')
	np.save('Cloud_free/MOD_CF_'+d1.strftime('%Y%m%d'),refl_vi)