#!/data/apps/anaconda/2.7-4.3.1/bin/python

from ClearCloud import ClearCloud
from glob import glob
import numpy as np
import scipy.io
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from skimage import io

t_mid = date(2013,8,10)
t_start = date(2013,8,2)
t_end = date(2013,8,17)

ndays = (t_end-t_start).days

oyscacld = np.ones((ndays*2,410,328))
count = 0
for i in range(ndays):
	for pre in ['MOD','MYD']:
		t = (t_start+timedelta(days=i)).strftime('%Y%m%d')
		res_files = glob('results/'+pre+'_'+t+'*.tif')
		if not res_files:
			count += 1
			continue
		elif len(res_files) == 1:
			res_file = res_files[0]
		else:
			res_file = [x for x in res_files if 'new' in x][0]
		mod_arr = io.imread(res_file)
		mod_arr[mod_arr==1]=2
		mod_arr[mod_arr==-1]=1
		oyscacld[count,:,:] = mod_arr
		count += 1

#L = scipy.ndimage.zoom(oyscacld,(1,2,2),order=0)

refl_vi = ClearCloud(oyscacld, t_start, t_end)

scipy.io.savemat('refl_in'+t_mid.strftime('%Y%m%d')+'.mat',mdict={'refl_in'+t_mid.strftime('%Y%m%d'):oyscacld})
scipy.io.savemat('refl_vi'+t_mid.strftime('%Y%m%d')+'.mat',mdict={'refl_vi'+t_mid.strftime('%Y%m%d'):refl_vi})