#!/data/apps/enthought_python/2.7.3/bin/python

import os
import numpy as np
import urllib, urllib2
from pyhdf.SD import SD, SDC
from datetime import datetime, date, timedelta as td

def latlon2pixzone(xll, yll, dx, dy, lat0,lat1,lon0,lon1):
	rl = abs((yll - lat0)/dy)
	ru = abs((yll - lat1)/dy)
	cu = abs((lon0 - xll)/dx)
	cl = abs((lon1 - xll)/dx)
	return int(round(rl)), int(round(ru)), int(round(cl)), int(round(cu))
	
def GetMOD10C1(yyyy0, mm0, dd0, yyyy1, mm1, dd1):
	dn0 = date(yyyy0, mm0, dd0)
	dn1 = date(yyyy1, mm1, dd1)
	dateDelta = dn1 - dn0
	ftp_path = 'ftp://n5eil01u.ecs.nsidc.org'
	[rl, ru, cl, cu] = latlon2pixzone(-180+0.05/2, 90-0.05/2, 0.05, -0.05, 60,25, -95,-140)
	camydsca = np.zeros(shape=(dateDelta.days + 1, ru-rl+1, cu-cl+1))
	camydcld = np.ones(shape=(dateDelta.days + 1, ru-rl+1, cu-cl+1))
	camodsca = np.zeros(shape=(dateDelta.days + 1, ru-rl+1, cu-cl+1))
	camodcld = np.ones(shape=(dateDelta.days + 1, ru-rl+1, cu-cl+1))
	for i in range(dateDelta.days + 1):
		ids = datetime.strftime(dn0 + td(days=i), '%Y.%m.%d')
		print '========'+ids+'========='
		myddir = '/SAN/MOSA/MYD10C1.006/' + ids
		moddir = '/SAN/MOST/MOD10C1.006/' + ids
		try:
			list_files = urllib.urlopen(ftp_path+myddir).read().splitlines()
			hdf_file1 = [x for x in list_files if ('.hdf' in x and '.xml' not in x)]
			hdf_file = hdf_file1[0].split(' ')[-1]
			f = urllib2.urlopen(ftp_path+myddir+'/'+hdf_file)
			with open('tempdata/'+hdf_file, "wb") as code:
				code.write(f.read())
			print 'download complete'
			hf = SD('tempdata/'+hdf_file, SDC.READ)
			mydsca = hf.select('Day_CMG_Snow_Cover')
			mydcld = hf.select('Day_CMG_Cloud_Obscured')
			camydsca[i] = mydsca[rl:ru+1, cl:cu+1]
			camydcld[i]	= mydcld[rl:ru+1, cl:cu+1]
			os.remove('tempdata/'+hdf_file)
		except:
			print myddir
			pass
		try:
			list_files1 = urllib.urlopen(ftp_path+moddir).read().splitlines()
			hdf_file2 = [x for x in list_files1 if ('.hdf' in x and '.xml' not in x)]
			hdf_file_mod = hdf_file2[0].split(' ')[-1]
			f1 = urllib2.urlopen(ftp_path+moddir+'/'+hdf_file_mod)
			with open('tempdata/'+hdf_file_mod, "wb") as code:
				code.write(f1.read())
			print 'download complete'
			hf1 = SD('tempdata/'+hdf_file_mod, SDC.READ)
			modsca = hf1.select('Day_CMG_Snow_Cover')
			modcld = hf1.select('Day_CMG_Cloud_Obscured')
			camodsca[i] = modsca[rl:ru+1, cl:cu+1]
			camodcld[i]	= modcld[rl:ru+1, cl:cu+1]
			os.remove('tempdata/'+hdf_file_mod)
		except:
			print moddir
			pass
	return camydsca, camydcld, camodsca, camodcld