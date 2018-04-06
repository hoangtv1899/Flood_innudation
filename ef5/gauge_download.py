#!/data/apps/anaconda/2.7-4.3.1/bin/python

import os
import numpy as np
import gdal
import urllib, urllib2
from datetime import datetime, timedelta
from dateutil import parser
import pytz
import xml.etree.ElementTree as ET
import pandas as pd
from osgeo import ogr
from osgeo import osr
from glob import glob
import shapely.geometry
import fiona

def pointSatisfy(long,lat):
	ycoor = int(abs(lat-geom[3])/geom[1])
	xcoor = int(abs(long-geom[0])/geom[1])
	if facArr[ycoor,xcoor] > 300:
		siteDF.loc[(siteDF.dec_lat_va == lat) &\
				(siteDF.dec_long_va == long),'basin_area'] = facArr[ycoor,xcoor]
		return 1
	else:
		return 0

def adjustCoor(long,lat):
	ycoor = int(abs(lat-geom[3])/geom[1])
	xcoor = int(abs(long-geom[0])/geom[1])
	miniArr = facArr[ycoor-1:ycoor+2,xcoor-1:xcoor+2]
	maxNum = np.max(miniArr)
	if maxNum < 250 or maxNum == facArr[ycoor,xcoor]:
		return 0
	a,b = np.where(miniArr==maxNum)
	siteDF.loc[(siteDF.dec_lat_va == lat) &\
				(siteDF.dec_long_va == long),'dec_lat_va'] = lat-(a[0]-1)*geom[1]
	siteDF.loc[(siteDF.dec_lat_va == lat) &\
				(siteDF.dec_long_va == long),'dec_long_va'] = long-(b[0]-1)*geom[1]
	siteDF.loc[(siteDF.dec_lat_va == lat) &\
				(siteDF.dec_long_va == long),'basin_area'] = maxNum
	return 1

def downloadDischarge(file_path):
	site_no1 = file_path.split('site_no=')[1].split('&')[0]
	if os.path.isfile('ef5/examples/obs/'+site_no1+'_discharge.csv'):
		return
	content1 = urllib2.urlopen(file_path).readlines()
	content1 = [x for x in content1 if '#' not in x]
	if len(content1) < 100:
		return
	header1 = content1[0].replace('\n','').split('\t')
	with open('ef5/examples/obs/'+site_no1+'_discharge.csv','w') as fo:
		fo.write('Date,Discharge(m3/s)\n')
		for line in content1[2:]:
			datetime_idx1 = header1.index('datetime')
			discharge_idx1 = [i for i,x in enumerate(header1) if '00060' in x.split('_')][0]
			line = line.replace('\n','').split('\t')
			#convert discharge from cfs to cms
			if line[discharge_idx1]:
				discharge1 = float(line[discharge_idx1])*0.028316847
				fo.write(datetime.strptime(line[datetime_idx1],'%Y-%m-%d').strftime('%Y-%m-%d %H:%M')+
							','+str(discharge1)+'\n')
	return 'gauge/'+site_no1+'_discharge.csv'

def xml2df(xml_data):
	root = ET.XML(xml_data) # element tree
	all_records = []
	for i, child in enumerate(root):
		record = {}
		for subchild in child:
			record[subchild.tag] = subchild.text
			if record not in all_records:
				all_records.append(record)
	return pd.DataFrame(all_records)

#read fac file
ds = gdal.Open('examples/basic/fac.tif')
geom = ds.GetGeoTransform()
facArr = ds.ReadAsArray()

#Read data from usgs stations
stationResPath = 'https://nwis.waterdata.usgs.gov/nwis/inventory?nw_longitude_va=-97.3587&nw_latitude_va=47.775&se_longitude_va=-85.8994&se_latitude_va=38.6911&coordinate_format=decimal_degrees&data_type=peak&group_key=NONE&format=sitefile_output&sitefile_output_format=xml&column_name=site_no&column_name=station_nm&column_name=dec_lat_va&column_name=dec_long_va&column_name=peak_begin_date&column_name=peak_end_date&list_of_search_criteria=lat_long_bounding_box%2Cdata_type'
content = urllib2.urlopen(stationResPath)
siteDF = xml2df(content.read())
siteDF['dec_lat_va'] = pd.to_numeric(siteDF.dec_lat_va)
siteDF['dec_long_va'] = pd.to_numeric(siteDF.dec_long_va)
siteDF['peak_begin_date'] = pd.to_datetime(siteDF.peak_begin_date)
siteDF['peak_end_date'] = pd.to_datetime(siteDF.peak_end_date)
siteDF['basin_area'] = np.nan
#for mississippi river
ss = pd.read_csv('ef5/selected_stations.csv')
siteno = ss.SITENO.tolist()
siteno = ['0'+str(x) for x in siteno]
selectSite = siteDF[siteDF.site_no.isin(siteno)]

#select stations inside domain
listP = [shapely.geometry.Point(xy) for xy in zip(siteDF.dec_long_va, siteDF.dec_lat_va)]
fc = fiona.open('fort_cobb.shp')
shp_record = fc.next()
domain = shapely.geometry.asShape(shp_record['geometry'])
selectSite = []
for i,p in enumerate(listP):
	if domain.contains(p):
		if pointSatisfy(p.x,p.y)==0:
			if adjustCoor(p.x,p.y)!=0:
				selectSite.append(i)
		else:
			selectSite.append(i)

tempDF = siteDF.iloc[selectSite]
finalSite = tempDF[tempDF['peak_end_date'] > datetime(2013,12,31)]

for site_no in finalSite.site_no.tolist():
	filePath = 'https://nwis.waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no='+site_no+'&referred_module=sw&period=&begin_date=2003-01-01&end_date=2013-12-31'
	downloadDischarge(filePath)

list_gauges = glob('gauge/*.csv')
list_gauges = [os.path.basename(x).split('_')[0] for x in list_gauges]
finalSite0 = finalSite[finalSite.site_no.isin(list_gauges)]
lats = finalSite0.dec_lat_va.tolist()
lons = finalSite0.dec_long_va.tolist()
siteName = finalSite0.station_nm.tolist()
siteNo = finalSite0.site_no.tolist()
finalSite0.to_csv('final_site.csv',index=False)	
driver = ogr.GetDriverByName("ESRI Shapefile")

# create the data source
data_source = driver.CreateDataSource("river_stations.shp")

# create the layer
srs = osr.SpatialReference()
layer = data_source.CreateLayer("river_stations", srs, ogr.wkbPoint)

# Add the fields we're interested in
field_region = ogr.FieldDefn("Station", ogr.OFTString)
field_region.SetWidth(24)
layer.CreateField(field_region)
site_region = ogr.FieldDefn("Site No", ogr.OFTString)
site_region.SetWidth(24)
layer.CreateField(site_region)
layer.CreateField(ogr.FieldDefn("Latitude", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("Longitude", ogr.OFTReal))

# Process the text file and add the attributes and features to the shapefile
for i,lat in enumerate(lats):
	# create the feature
	feature = ogr.Feature(layer.GetLayerDefn())
	# Set the attributes using the values from the delimited text file
	feature.SetField("Station", siteName[i])
	feature.SetField("Site No", siteNo[i])
	feature.SetField("Latitude", float(lat))
	feature.SetField("Longitude", float(lons[i]))
	
	# create the WKT for the feature using Python string formatting
	wkt = "POINT(%f %f)" %  (float(lons[i]) , float(lat))
	
	# Create the point from the Well Known Txt
	point = ogr.CreateGeometryFromWkt(wkt)
	
	# Set the feature geometry using the point
	feature.SetGeometry(point)
	# Create the feature in the layer (shapefile)
	layer.CreateFeature(feature)
	# Dereference the feature
	feature = None

# Save and close the data source
data_source = None

