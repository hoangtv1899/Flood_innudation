#!/data/apps/anaconda/2.7-4.3.1/bin/python

import os
import numpy as np
import gdal
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from glob import glob

#Landsat 7 satellite
land7_files = sorted(glob('ref_satellite/LE07_*.[Tt][Ii][Ff]'))
for l7 in land7_files:
	fname7 = os.path.basename(l7)
	date7 = datetime.strptime(fname7.split('_')[3],'%Y%m%d')
	b7 = fname7.split('.')[0].split('_')[-1]
	newFile7 = 'ref_satellite/reprojected/LE07_'+date7.strftime('%Y%m%d')+'_'+b7+'.tif'
	cmd7 = 'gdalwarp -s_srs EPSG:32616 -t_srs EPSG:32615 -te 727194.309497 4131153.827477 820194.309497 4281903.827477 '+l7+' '+newFile7
	os.system(cmd7)

#Landsat 8 satellite
land8_files = sorted(glob('ref_satellite/LC08_*.[Tt][Ii][Ff]'))
for l8 in land8_files:
	fname8 = os.path.basename(l8)
	date8 = datetime.strptime(fname8.split('_')[3],'%Y%m%d')
	b8 = fname8.split('.')[0].split('_')[-1]
	newFile8 = 'ref_satellite/reprojected/LC08_'+date8.strftime('%Y%m%d')+'_'+b8+'.tif'
	cmd8 = 'gdalwarp -s_srs EPSG:32616 -t_srs EPSG:32615 -te 727194.309497 4131153.827477 820194.309497 4281903.827477 '+l8+' '+newFile8
	os.system(cmd8)

#IRS AWiFS satellite
irs_files = sorted(glob('ref_satellite/R2AWF*.[Tt][Ii][Ff]'))
for irs_file in irs_files:
	irs_name = os.path.basename(irs_file)
	irs_date = datetime.strptime(irs_name.split('R2AWF')[1][:8],'%m%d%Y')
	irs_b = irs_name.split('_')[1]
	irs_newfile = 'ref_satellite/reprojected/R2AWF_'+irs_date.strftime('%Y%m%d')+'_'+irs_b+'.tif'
	irs_cmd = 'gdalwarp -s_srs \'+proj=lcc +lat_1=41.51344134676314 +lat_2=36.97182770229064 +lat_0=39.28448575702283 +lon_0=-89.519593949697 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs\' -t_srs EPSG:32615 -te 727194.309497 4131153.827477 820194.309497 4281903.827477 '+irs_file+' '+irs_newfile
	os.system(irs_cmd)
