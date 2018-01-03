#!/data/apps/enthought_python/2.7.3/bin/python

import os
import numpy as np
from cookielib import CookieJar
from urllib import urlencode
from pyhdf.SD import SD, SDC
from datetime import datetime, date, timedelta as td
import urllib2
from glob import glob

def GetMOD09GQ(dn0, dn1, tiles):
	username = 'hoangtv1989'
	password = 'Chrs2017'
	
	password_manager = urllib2.HTTPPasswordMgrWithDefaultRealm()
	password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)
	
	cookie_jar = CookieJar() 
	# Install all the handlers.
	 
	opener = urllib2.build_opener(
		urllib2.HTTPBasicAuthHandler(password_manager),
		#urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
		#urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
		urllib2.HTTPCookieProcessor(cookie_jar))
	urllib2.install_opener(opener)
	
	dateDelta = dn1 - dn0
	
	ftp_path = 'https://e4ftl01.cr.usgs.gov/'
	tempdata_path = 'tempdata/'#'/share/ssd-scratch/htranvie/Flood/tempdata/'
	list_hdf = []
	for i in range(dateDelta.days+1):
		ids = datetime.strftime(dn0 + td(days=i), '%Y.%m.%d')
		print '========'+ids+'========='
		for dir in ['MOLA/MYD09GQ.006/','MOLT/MOD09GQ.006/']:
			moddir = dir+ids
			try:
				request = urllib2.Request(ftp_path+moddir)
				response = urllib2.urlopen(request)
			except:
				print 'error '+dir+' '+ids
				continue
			list_files = response.read().splitlines()
			for tile in tiles:
				tile_files = [x.split('href="')[1].split('">')[0] for x in list_files if tile in x]
				try:
					hdf_file = [x for x in tile_files if os.path.splitext(x)[1]=='.hdf'][0]
				except:
					print 'error '+dir+' '+ids
					continue
				if not os.path.isfile(tempdata_path+hdf_file):
					f = urllib2.urlopen(ftp_path+moddir+'/'+hdf_file)
					with open(tempdata_path+hdf_file, "wb") as code:
						code.write(f.read())
				print 'download complete'
		list_hdf += glob('/share/ssd-scratch/htranvie/Flood/tempdata/*'+(dn0+td(days=i)).strftime('%Y%j')+'*.hdf')
	#copy the hdf file to typhoon for reprojection
	os.system('scp '+' '.join(list_hdf)+' pconnect@typhoon.eng.uci.edu:/mnt/t/disk2/pconnect/CHRSData/python/modis/tempdata/')
	#reprojection on typhoon server
	os.system('ssh pconnect@typhoon.eng.uci.edu /mnt/t/disk2/pconnect/CHRSData/python/modis/reprojection.py '+dn0.strftime('%Y%m%j')+' '+dn1.strftime('%Y%m%j')+' mississippi_rec1')
	#copy back the tiff file to hpc
	os.system('scp pconnect@typhoon.eng.uci.edu:/mnt/t/disk2/pconnect/CHRSData/python/modis/clipped_data/*.tif  /share/ssd-scratch/htranvie/Flood/data/geotiff/')
	

def GetMOD09GA(dn0, dn1, tile):
	username = 'hoangtv1989'
	password = 'Chrs2017'
	
	password_manager = urllib2.HTTPPasswordMgrWithDefaultRealm()
	password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)
	
	cookie_jar = CookieJar() 
	# Install all the handlers.
	 
	opener = urllib2.build_opener(
		urllib2.HTTPBasicAuthHandler(password_manager),
		#urllib2.HTTPHandler(debuglevel=1),    # Uncomment these two lines to see
		#urllib2.HTTPSHandler(debuglevel=1),   # details of the requests/responses
		urllib2.HTTPCookieProcessor(cookie_jar))
	urllib2.install_opener(opener)
	
	dateDelta = dn1 - dn0
	
	ftp_path = 'https://e4ftl01.cr.usgs.gov/'
	list_hdf = []
	for i in range(dateDelta.days+1):
		ids = datetime.strftime(dn0 + td(days=i), '%Y.%m.%d')
		print '========'+ids+'========='
		for dir in ['MOLT/MOD09GA.006/',
					'MOLA/MYD09GA.006/']:
			moddir = dir+ids
			try:
				request = urllib2.Request(ftp_path+moddir)
				response = urllib2.urlopen(request)
			except:
				print 'error '+dir+' '+ids
				continue
			list_files = response.read().splitlines()
			tile_files = [x.split('href="')[1].split('">')[0] for x in list_files if tile in x]
			try:
				hdf_file = [x for x in tile_files if os.path.splitext(x)[1]=='.hdf'][0]
			except:
				print 'error '+dir+' '+ids
				continue
			if not os.path.isfile('/share/ssd-scratch/htranvie/Flood/tempdata/'+hdf_file):
				f = urllib2.urlopen(ftp_path+moddir+'/'+hdf_file)
				with open('/share/ssd-scratch/htranvie/Flood/tempdata/'+hdf_file, "wb") as code:
					code.write(f.read())
			list_hdf.append('/share/ssd-scratch/htranvie/Flood/tempdata/'+hdf_file)
			print 'download complete'
	os.system('scp '+' '.join(list_hdf)+' pconnect@typhoon.eng.uci.edu:/mnt/t/disk2/pconnect/CHRSData/python/modis/tempdata/')
	#reprojection on typhoon server
	os.system('ssh pconnect@typhoon.eng.uci.edu /mnt/t/disk2/pconnect/CHRSData/python/modis/reprojection_band3.sh')
	#copy back the tiff file to hpc
	os.system('scp pconnect@typhoon.eng.uci.edu:"'+
				' '.join(['/mnt/t/disk2/pconnect/CHRSData/python/modis/clipped_data/'+x.split('/')[-1].split('.006')[0]+'_b*.tif' for x in list_hdf])+'" /share/ssd-scratch/htranvie/Flood/data/geotiff/')