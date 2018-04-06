#!/data/apps/anaconda/2.7-4.3.1/bin/python

import os
import numpy as np
import gdal
import urllib, urllib2
from datetime import datetime, timedelta
from dateutil import parser
import pytz

def downloadDischarge(file_path):
	content1 = urllib2.urlopen(file_path).readlines()
	site_no1 = file_path.split('site_no=')[1].split('&')[0]
	content1 = [x for x in content1 if '#' not in x]
	header1 = content1[0].replace('\n','').split('\t')
	with open('gauge/'+site_no1+'_discharge.csv','w') as fo:
		for line in content1[2:]:
			datetime_idx1 = header1.index('datetime')
			tz_idx1 = header1.index('tz_cd')
			discharge_idx1 = [i for i,x in enumerate(header1) if x.split('_')[-1] == '00060'][0]
			line = line.replace('\n','').split('\t')
			#convert discharge from cfs to cms
			discharge1 = float(line[discharge_idx1])*0.028316847
			fo.write(line[datetime_idx1]+' '+line[tz_idx1]+'\t'+str(discharge1)+'\n')
	return 'gauge/'+site_no1+'_discharge.csv'

def downloadHeight(file_path, bed_elev):
	content1 = urllib2.urlopen(file_path).readlines()
	site_no1 = file_path.split('site_no=')[1].split('&')[0]
	content1 = [x for x in content1 if '#' not in x]
	header1 = content1[0].replace('\n','').split('\t')
	with open('gauge/'+site_no1+'_height.csv','w') as fo:
		for line in content1[2:]:
			datetime_idx1 = header1.index('datetime')
			tz_idx1 = header1.index('tz_cd')
			height_idx1 = [i for i,x in enumerate(header1) if x.split('_')[-1] == '00065'][0]
			line = line.replace('\n','').split('\t')
			if line[height_idx1]:
				#convert height from feet to meter
				height1 = (bed_elev+float(line[height_idx1]))*0.3048
				fo.write(line[datetime_idx1]+' '+line[tz_idx1]+'\t'+str(height1)+'\n')
	return 'gauge/'+site_no1+'_height.csv'

#read dem data high resolution
ds_dem0 = gdal.Open('gmted_mean075_reprojected.tif')
geom0 = ds_dem0.GetGeoTransform()
dem_arr0 = ds_dem0.ReadAsArray()
header0 = "ncols     %s\n" % dem_arr0.shape[1]
header0 += "nrows    %s\n" % dem_arr0.shape[0]
header0 += "xllcorner %.3f\n" % geom0[0]
header0 += "yllcorner %.3f\n" % (geom0[3]+geom0[5]*dem_arr0.shape[0])
header0 += "cellsize %.2f\n" % geom0[1]
header0 += "NODATA_value -9999\n"

with open('lower_mississippi_high_res.dem.asc','w') as fo:
	fo.write(header0)
	np.savetxt(fo, dem_arr0, fmt="%d")

#read dem data
ds_dem = gdal.Open('gmted_mean075_reprojected_resampled.tif')
geom = ds_dem.GetGeoTransform()
dem_arr = ds_dem.ReadAsArray()
dem_arr_river = gdal.Open('gmted_river_resampled_bilinear.tif').ReadAsArray()
n_river = np.logical_and(dem_arr_river!=-99,bin_water>0)
dem_arr[n_river] = dem_arr_river[n_river]

header = "ncols     %s\n" % dem_arr.shape[1]
header += "nrows    %s\n" % dem_arr.shape[0]
header += "xllcorner %.3f\n" % geom[0]
header += "yllcorner %.3f\n" % (geom[3]+geom[5]*dem_arr.shape[0])
header += "cellsize %.2f\n" % geom[1]
header += "NODATA_value -9999\n"

with open('lower_mississippi.dem.asc','w') as fo:
	fo.write(header)
	np.savetxt(fo, dem_arr, fmt="%d")

#create width ascii data
bin_water = gdal.Open('swbd_lower_mississippi_resampled.tif').ReadAsArray()
small_fix = gdal.Open('ttt.tif').ReadAsArray()
inter_value = gdal.Open('swbd_raster_resampled.tif').ReadAsArray()

bin_water[np.logical_and(small_fix==1,bin_water==0)]=1
bin_water[np.logical_and(small_fix==0,bin_water==1)]=0

bin_water[np.logical_and(bin_water==1,inter_value < 0.3)] = 750/3.
bin_water[np.logical_and(bin_water==1,np.logical_and(inter_value >= 0.3, inter_value < 0.7))] = 2*750/3.
bin_water[np.logical_and(bin_water==1,bin_water>=0.7)] = 750

with open('lower_mississippi.width.asc','w') as fo1:
	fo1.write(header)
	np.savetxt(fo1, bin_water,fmt='%.2f')

temp_water_depth = np.loadtxt('res_SGC-0013.wd',skiprows=6)
temp_water_depth[bin_water==0] = 0
#find water pixel that has not been given depth:
rem_water_depth = np.logical_and(temp_water_depth==0,bin_water>0)
#find depth water pixel which has same elevation and same width
rx, ry = np.where(rem_water_depth)
for i,x in enumerate(rx):
	elev = dem_arr[x,ry[i]]
	wx, wy = np.where(np.logical_and(dem_arr==elev,temp_water_depth!=0))
	if wx.size:
		width = bin_water[x,ry[i]]
		for j,wxi in enumerate(wx):
			if bin_water[wxi,wy[j]] == width:
				temp_water_depth[x,ry[i]] = temp_water_depth[wxi,wy[j]]

temp_water_depth[rx[np.logical_and(rx<70,np.logical_and(ry>50,ry<75))],
				ry[np.logical_and(rx<70,np.logical_and(ry>50,ry<75))]] = 8*0.3048
temp_water_depth[rx[np.logical_and(rx>100,np.logical_and(ry>100,ry<126))],
				ry[np.logical_and(rx>100,np.logical_and(ry>100,ry<126))]] = 39*0.3048

with open('lower_mississippi.depth.asc','w') as fo1:
	fo1.write(header)
	np.savetxt(fo1, temp_water_depth,fmt='%.2f')

#Read data from usgs stations
#Station mississippi river at st.louis in November 2015
file_path1 = 'https://nwis.waterdata.usgs.gov/mo/nwis/uv/?cb_00060=on&cb_00065=on&format=rdb&site_no=07010000&period=&begin_date=2015-11-01&end_date=2016-02-29'
#dis_file1 = downloadDischarge(file_path1)
h_file1 = downloadHeight(file_path1,379.53)
#Station mississippi river at chester in November 2015
file_path2 = 'https://nwis.waterdata.usgs.gov/mo/nwis/uv/?cb_00060=on&cb_00065=on&format=rdb&site_no=07020500&period=&begin_date=2015-11-01&end_date=2016-02-29'
#dis_file2 = downloadDischarge(file_path2)
h_file2 = downloadHeight(file_path2,340.51)
#Station mississippi river at cape girardeau in November 2015
file_path3 = 'https://nwis.waterdata.usgs.gov/mo/nwis/uv/?cb_00065=on&format=rdb&site_no=07020850&period=&begin_date=2015-11-01&end_date=2016-02-29'
h_file3 = downloadHeight(file_path3,304.65)

#create bdy file for lisflood
gauge_files = [h_file1,h_file2,h_file3]
timezone = pytz.timezone('America/Chicago')
t_start = timezone.localize(datetime(2015,11,1,0))
with open('bdy_file.bdy','w') as bdy_file:
	bdy_file.write('# '+datetime.now().strftime('%Y-%m-%d %H:%M')+'\n')
	for file in gauge_files:
		station_name = os.path.basename(file).split('_')[0]
		content = open(file,'r').readlines()
		bdy_file.write('station_'+station_name+'\n')
		bdy_file.write('\t'+str(len(content))+'\t'+'seconds\n')
		for line in content:
			line = line.replace('\n','')
			curr_t = timezone.localize(parser.parse(line.split('\t')[0]))
			elapsed_seconds = int((curr_t-t_start).total_seconds())
			if elapsed_seconds >=0:
				bdy_file.write('\t'+line.split('\t')[1]+'\t'+str(elapsed_seconds)+'\n')