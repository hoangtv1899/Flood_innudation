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

d0 = date(2013,6,8)
for i in range(1,190):
	d1 = d0 + timedelta(days=i)
	if os.path.isfile('Cloud_free/MOD_CF_'+d1.strftime('%Y%m%d')):
		continue
	selectedDays = [(d1+timedelta(days=-x-1)).strftime('%Y%m%d') for x in range(7)][::-1]+\
					[(d1+timedelta(days=x)).strftime('%Y%m%d') for x in range(8)]
	refl_in = np.ones((15,95,95))*-1
	count = 0
	for j,day in enumerate(selectedDays):
		if os.path.isfile('Cloud_free/MOD_CF_'+d1.strftime('%Y%m%d')+'.npy'):
			arr = np.load('Cloud_free/MOD_CF_'+d1.strftime('%Y%m%d')+'.npy')
		elif os.path.isfile('results/clipped/MOD_'+day+'_bin.tif'):
			arr = io.imread('results/clipped/MOD_'+day+'_bin.tif')
		else:
			count += 1
			continue
		refl_in[j,:,:] = arr
	if count > 7:
		continue
	refl_in[refl_in==1] = 2
	refl_in[refl_in==-1] = 1
	#L = scipy.ndimage.zoom(refl_in,(1,2,2),order=1)
	
	refl_vi = ClearCloud(refl_in, d1+timedelta(days=-7), d1+timedelta(days=7))
	plt.clf()
	plt.imshow(refl_vi)
	plt.savefig('Cloud_free/img/h'+d1.strftime('%Y%m%d')+'.png')
	np.save('Cloud_free/MOD_CF_'+d1.strftime('%Y%m%d'),refl_vi)