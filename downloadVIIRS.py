#!/data/apps/enthought_python/2.7.3/bin/python

import os
import shutil
import numpy as np
import urllib, urllib2
from osgeo import gdal, gdalnumeric, ogr
from glob import glob
from datetime import datetime, timedelta
import tarfile
import gzip
import h5py
import multiprocessing as mp

def h5toGTiff(data_h5, geom_h5, xs, ys, outfile):
	viirs_filename = os.path.basename(data_h5).split('.')[0]
	lat_granule = 'HDF5:"%s"://All_Data/VIIRS-IMG-GEO-TC_All/Latitude' %geom_h5
	lon_granule = 'HDF5:"%s"://All_Data/VIIRS-IMG-GEO-TC_All/Longitude' %geom_h5
	refl_granule = 'HDF5:"%s"://All_Data/VIIRS-I1-SDR_All/Reflectance' %data_h5
	
	os.system('gdal_translate -of VRT %s /share/ssd-scratch/htranvie/Flood/tempdata/%s_lat.vrt' %( lat_granule, viirs_filename))
	os.system('gdal_translate -of VRT %s /share/ssd-scratch/htranvie/Flood/tempdata/%s_lon.vrt' %( lon_granule, viirs_filename))
	
	refl_vrt="""<VRTDataset rasterXSize="%d" rasterYSize="%d">
	  <SRS>GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0,AUTHORITY[&quot;EPSG&quot;,&quot;8901&quot;]],UNIT[&quot;degree&quot;,0.0174532925199433,AUTHORITY[&quot;EPSG&quot;,&quot;9108&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]</SRS>
	   <Metadata domain="GEOLOCATION">
		 <MDI key="X_DATASET">/share/ssd-scratch/htranvie/Flood/tempdata/%s_lon.vrt</MDI>
		 <MDI key="X_BAND">1</MDI>
		 <MDI key="Y_DATASET">/share/ssd-scratch/htranvie/Flood/tempdata/%s_lat.vrt</MDI>
		 <MDI key="Y_BAND">1</MDI>
		 <MDI key="PIXEL_OFFSET">0</MDI>
		 <MDI key="LINE_OFFSET">0</MDI>
		 <MDI key="PIXEL_STEP">1</MDI>
		 <MDI key="LINE_STEP">1</MDI>
	   </Metadata> 
	  <VRTRasterBand dataType="UInt16" band="1">
		<Metadata />
		<SimpleSource>
		  <SourceFilename relativeToVRT="0">%s</SourceFilename>
		  <SourceBand>1</SourceBand>
		  <SourceProperties RasterXSize="%d" RasterYSize="%d" DataType="UInt16" BlockXSize="6400" BlockYSize="1536" />
		  <SrcRect xOff="0" yOff="0" xSize="%d" ySize="%d" />
		  <DstRect xOff="0" yOff="0" xSize="%d" ySize="%d" />
		</SimpleSource>
	  </VRTRasterBand>
	</VRTDataset>""" %( xs, ys, viirs_filename, viirs_filename, \
						refl_granule, xs, ys, xs, ys, xs, ys)
	
	with open('/share/ssd-scratch/htranvie/Flood/tempdata/%s.vrt' %viirs_filename ,'w') as fo:
		fo.write(refl_vrt)
	
	os.system('gdalwarp -of GTiff -co "COMPRESS=LZW" \
				/share/ssd-scratch/htranvie/Flood/tempdata/%s.vrt '%viirs_filename +outfile)


def UnzipNCheck(geom_h5):
	geom_noninter = []
	geom_inter = []
	
	geom_h5_uz = geom_h5[:-3]
	if not os.path.isfile(geom_h5_uz):
		with open(geom_h5_uz,'w') as of:
			try:
				of.write(gzip.GzipFile(fileobj=file(geom_h5),mode='rb').read())
			except:
				return
	try:
		geom_hf = h5py.File(geom_h5_uz,'r')
	except:
		os.remove(geom_h5_uz)
		return
	lat = geom_hf['All_Data']['VIIRS-IMG-GEO-TC_All']['Latitude'][:]
	lon = geom_hf['All_Data']['VIIRS-IMG-GEO-TC_All']['Longitude'][:]
	xmin, xmax, ymin, ymax = lon.min(), lon.max(), lat.min(), lat.max()
	if (xmin > exts[1]) or (xmax < exts[0]) or (ymin > exts[3]) or (ymax < exts[2]) \
		or (abs(xmax - xmin) >= 350):
		geom_noninter.append(geom_h5.split('GITCO_')[1].split('_c')[0])
	else:
		geom_inter.append(geom_h5.split('GITCO_')[1].split('_c')[0])
	
	return geom_inter, geom_noninter

def UnzipDat(dat_h5):
	dat_h5_uz = dat_h5[:-3]
	if not os.path.isfile(dat_h5_uz):
		with open(dat_h5_uz,'w') as of:
			try:
				of.write(gzip.GzipFile(fileobj=file(dat_h5),mode='rb').read())
			except:
				return
	
def init(exts1):
	global exts
	exts = exts1

def init1(geom_inter1):
	global geom_inter
	geom_inter = geom_inter1 	

def Untar(geotar):
	dest_geodir = geotar.split('.')[0]
	try:
		geom_tar = tarfile.open(mode="r:tar",fileobj=file(geotar))
	except:
		return
	if os.path.isdir(dest_geodir):
		ab = [os.path.basename(x) for x in glob(geotar.split('.')[0]+'/*.gz')]
		if ab == geom_tar.getnames():
			return
		else:
			shutil.rmtree(dest_geodir)
	os.mkdir(dest_geodir)
	geom_tar.extractall(dest_geodir)
	geom_tar.close()

def UntarPart(datatar):
	dest_datadir = datatar.split('.')[0]
	try:
		dat_tar = tarfile.open(mode="r:tar",fileobj=file(datatar))
	except:
		return
	if os.path.isdir(dest_datadir):
		shutil.rmtree(dest_datadir)
	os.mkdir(dest_datadir)
	list_intar = dat_tar.getnames()
	for mem in list_intar:
		if mem.split('SVI01_')[1].split('_c')[0] in geom_inter:
			dat_tar.extractall(dest_datadir,members=[dat_tar.getmember(mem)])
	dat_tar.close()

#Read and extract extends from shapefiles
shp_file = 'shapes/missouri.shp'
#shp_raster = 'shapes/ex_watershed.tif'
shp = ogr.Open(shp_file)
shp_layer = shp.GetLayer()
shp_feat = shp_layer.GetFeature(0)
shp_geom = shp_feat.GetGeometryRef()
exts = shp_layer.GetExtent()

#create raster of the watershed
#os.system('gdal_rasterize -a LEVEL_1 -tr '+str(geotransform[1])+' '+str(geotransform[1])+' -l '+os.path.basename(shp_file).split('.')[0]+' '+shp_file+' '+shp_raster)
def findGit_VIIRS(yyyy1,mm1,dd1,exts):
	#Extraction day
	if mm1<10:
		mm1 = '0'+str(mm1)
	if dd1<10:
		dd1 = '0'+str(dd1)
	viirs_dir = str(yyyy1)+str(mm1)+str(dd1)
	#List date directory for ftp
	viirs_ftp_path = 'ftp://ftp-npp.class.ngdc.noaa.gov/'
	print 'date to enter: '+viirs_dir
	i_gitfile = '/VIIRS-SDR/VIIRS-Image-Bands-SDR-Ellipsoid-Terrain-Corrected-Geo/'
	i_datafile = '/VIIRS-SDR/VIIRS-Imagery-Band-01-SDR/'
	#list all terrain corrected geo tar and xml files
	list_gittar = sorted([x.split(' ')[-1] for x in \
				urllib.urlopen(viirs_ftp_path+viirs_dir+i_gitfile).read().splitlines() \
				if x.split('.')[-1] == 'tar'])
	list_gitxml = sorted([x.split(' ')[-1] for x in \
				urllib.urlopen(viirs_ftp_path+viirs_dir+i_gitfile).read().splitlines() \
				if '.xml' in x])
	print 'number of tar files: '+str(len(list_gittar))
	#loop through tar list one by one
	not_inter_orbit = []
	print 'start to download one by one'
	geom_inter = []
	for jk, gittar in enumerate(list_gittar):
		list_gith5_files = sorted([x.split('GITCO_')[1].split('_c')[0] for x in \
				urllib.urlopen(viirs_ftp_path+viirs_dir+i_gitfile+list_gitxml[jk]).readlines() \
				if '<FileName>' in x])
		#create a dictionary of orbit in the tar file
		orbit_dict = dict((x,[]) for x in set([x.split('_')[-1] for x in list_gith5_files]))
		for orbit in orbit_dict.keys():
			for x in list_gith5_files:
				if not_inter_orbit:
					list_not_inter = sorted([x1 for x1 in not_inter_orbit if orbit in x1])
					if list_not_inter:
						last_sc = list_not_inter[-1]
					else:
						last_sc = ''
				else:
					last_sc = ''
				if orbit in x:
					if last_sc:
						t_e = viirs_dir+last_sc.split('_e')[1].split('_')[0]+'000'
						t_x = viirs_dir+x.split('_e')[1].split('_')[0]+'000'
						if (datetime.strptime(t_x,'%Y%m%d%H%M%f') - \
							datetime.strptime(t_e,'%Y%m%d%H%M%f')).seconds < 18000:
							print 'orbit that not intersect the ROI '+x
							continue
					orbit_dict[orbit].append(x)
		if all(orbit_dict[x] == [] for x in orbit_dict.keys()):
			print 'skip '+gittar
			continue
		dest_gittar = '/share/ssd-scratch/htranvie/Flood/tempdata/'+gittar
		dest_gitdir = '/share/ssd-scratch/htranvie/Flood/tempdata/'+gittar.split('.')[0]
		if not os.path.isfile(dest_gittar):
			if not os.path.isdir(dest_gitdir):
				urllib.urlretrieve(viirs_ftp_path+viirs_dir+i_gitfile+gittar,\
								dest_gittar)
			else:
				continue
		#start to open the tar file
		geom_tar = tarfile.open(mode="r:tar",fileobj=file(dest_gittar))
		if not os.path.isdir(dest_gitdir):
			os.mkdir(dest_gitdir)
		list_geom_gz = geom_tar.getnames()
		xmin0, xmax0, ymin0, ymax0 = 0,0,0,0
		for orbit in orbit_dict.keys():
			print 'processing: '+orbit
			for sc in orbit_dict[orbit]:
				gz_in_tar = [x for x in list_geom_gz if sc in x][0]
				geom_tar.extractall(dest_gitdir,members=[geom_tar.getmember(gz_in_tar)])
				geom_h5_uz = dest_gitdir+'/'+gz_in_tar[:-3]
				if not os.path.isfile(geom_h5_uz):
					with open(geom_h5_uz,'w') as of:
						of.write(gzip.GzipFile(fileobj=file(dest_gitdir+'/'+gz_in_tar),mode='rb').read())
				try:
					geom_hf = h5py.File(geom_h5_uz,'r')
				except IOError:
					continue
				lat = geom_hf['All_Data']['VIIRS-IMG-GEO-TC_All']['Latitude'][:]
				lon = geom_hf['All_Data']['VIIRS-IMG-GEO-TC_All']['Longitude'][:]
				xmin, xmax, ymin, ymax = lon.min(), lon.max(), lat.min(), lat.max()
				if (xmin > exts[1]) or (xmax < exts[0]) \
					or (ymin > exts[3]) or (ymax < exts[2]) \
					or (abs(xmax - xmin) >= 350):
					if not all([xmin0, xmax0, ymin0, ymax0]):
						xmin0, xmax0, ymin0, ymax0 = xmin, xmax, ymin, ymax
					else:
						exts_centroid = (np.mean([exts[1],exts[0]]),np.mean([exts[3],exts[2]]))
						x0 = (np.mean([xmin0,xmax0]),np.mean([ymin0,ymax0]))
						x1 = (np.mean([xmin,xmax]),np.mean([ymin,ymax]))
						dt0 = np.linalg.norm(np.array(x0)-np.array(exts_centroid))
						dt1 = np.linalg.norm(np.array(x1)-np.array(exts_centroid))
						if dt1 > dt0:
							not_inter_orbit.append(sc)
							print 'skip this '+sc
							break
						elif (abs(xmax-xmin)>=350):
							not_inter_orbit.append(sc)
							print 'skip this '+sc
							break
						else:
							continue
				else:
					print 'accept this '+sc
					geom_inter.append(sc)
	for viirs_tarfile in glob('/share/ssd-scratch/htranvie/Flood/tempdata/VIIRS-SDR_VIIRS-Image-Bands-SDR-Ellipsoid-Terrain-Corrected-Geo_'+viirs_dir+'*.tar'):
		os.remove(viirs_tarfile)
	return geom_inter
			

def findData_VIIRS(geom_inter):
	viirs_dir_list = list(set([x.split('_d')[1].split('_')[0] for x in geom_inter]))
	#List date directory for ftp
	viirs_ftp_path = 'ftp://ftp-npp.class.ngdc.noaa.gov/'
	i_gitfile = '/VIIRS-SDR/VIIRS-Image-Bands-SDR-Ellipsoid-Terrain-Corrected-Geo/'
	i_datafile = '/VIIRS-SDR/VIIRS-Imagery-Band-01-SDR/'
	for viirs_dir in viirs_dir_list:
		if not os.path.isdir('/ssd-scratch/htranvie/Flood/data/'+viirs_dir):
			os.mkdir('/ssd-scratch/htranvie/Flood/data/'+viirs_dir)
		print 'date to enter: '+viirs_dir
		#list all data viirs band 1 xml files
		list_datxml = sorted([x.split(' ')[-1] for x in \
					urllib.urlopen(viirs_ftp_path+viirs_dir+i_datafile).read().splitlines() \
					if '.xml' in x])
		print 'number of xml files: '+str(len(list_datxml))
		for datxml in list_datxml:
			list_dath5_files = sorted([x.split('<FileName>')[1].split('</FileName>')[0] for x in \
				urllib.urlopen(viirs_ftp_path+viirs_dir+i_datafile+datxml).readlines() \
				if '<FileName>' in x])
			ROI_dath5_files = [x for x in list_dath5_files if \
								x.split('SVI01_')[1].split('_c')[0] in geom_inter]
			if not ROI_dath5_files:
				continue
			else:
				dattar = datxml.split('.manifest')[0]
				print dattar
				dest_dattar = '/share/ssd-scratch/htranvie/Flood/tempdata/'+dattar
				dest_datdir = '/share/ssd-scratch/htranvie/Flood/tempdata/'+dattar.split('.')[0]
				urllib.urlretrieve(viirs_ftp_path+viirs_dir+i_datafile+dattar,\
								dest_dattar)
				if not os.path.isdir(dest_datdir):
					os.mkdir(dest_datdir)
				dat_tar = tarfile.open(mode="r:tar",fileobj=file(dest_dattar))
				list_dat_gz = dat_tar.getnames()
				for dat_gz in list_dat_gz:
					if dat_gz[:-3] in ROI_dath5_files:
						dat_tar.extractall(dest_datdir,members=[dat_tar.getmember(dat_gz)])
						dat_h5_uz = dest_datdir+'/'+dat_gz[:-3]
						if not os.path.isfile(dat_h5_uz):
							with open(dat_h5_uz,'w') as of:
								of.write(gzip.GzipFile(fileobj=file(dest_datdir+'/'+dat_gz),mode='rb').read())
		for orbit in geom_inter:
			git_h5, dat_h5 = sorted(glob('/ssd-scratch/htranvie/Flood/tempdata/*/*'+orbit+'*.h5'))
			h5toGTiff(dat_h5, git_h5, 6400, 6144, '/ssd-scratch/htranvie/Flood/data/'+viirs_dir+'/SVI01_'+orbit+'.tif')
		
			
list_dest_gittar = glob('/share/ssd-scratch/htranvie/Flood/tempdata/*VIIRS-Image-Bands-SDR-Ellipsoid-Terrain*.tar')

pool = mp.Pool(initializer=init, initargs=(exts,), processes=64)
pool.map(Untar, list_dest_gittar)

geom_h5_gz_list = []
for gittar in list_dest_gittar:
	dest_gitdir = gittar.split('.')[0]
	geom_h5_gz_list += glob(dest_gitdir+'/*.gz')

result = pool.map(UnzipNCheck, geom_h5_gz_list)
geom_inter = []
geom_noninter = []
for res in result:
	geom_inter += res[0]
	geom_noninter += res[1]


list_dataxml = [x.split(' ')[-1] for x in \
				urllib.urlopen(viirs_ftp_path+viirs_dir+i_datafile).read().splitlines() \
				if '.xml' in x]

list_datatar = []
for dataxml in list_dataxml:
	datah5_list = [x.split('SVI01_')[1].split('_c')[0] for x in \
				urllib.urlopen(viirs_ftp_path+viirs_dir+i_datafile+dataxml).readlines() \
				if '<FileName>' in x]
	if np.sum([int(x in geom_noninter) for x in datah5_list]) < len(datah5_list):
		list_datatar.append(dataxml[:-13])

for datatar in list_datatar:
	dest_datatar = '/share/ssd-scratch/htranvie/Flood/tempdata/'+datatar
	if os.path.isfile(dest_datatar):
		continue
	else:
		urllib.urlretrieve(viirs_ftp_path+viirs_dir+i_datafile+datatar,\
							dest_datatar)

list_dest_datatar = glob('/share/ssd-scratch/htranvie/Flood/tempdata/*VIIRS-Imagery-Band-01-SDR*.tar')
pool = mp.Pool(initializer=init1, initargs=(geom_inter,), processes=64)
pool.map(UntarPart, list_dest_datatar)
dat_h5_gz_list = []
for geotar in list_dest_datatar:
	dest_geodir = geotar.split('.')[0]
	dat_h5_gz_list += glob(dest_geodir+'/*.gz')

pool = mp.Pool()
pool.map(UnzipDat, dat_h5_gz_list)

dat_h5_list = [x[:-3] for x in dat_h5_gz_list]
geo_h5_list = []
for file in geom_h5_gz_list:
	temp_name = file.split('GITCO_')[1].split('_c')[0]
	if temp_name in geom_inter:
		geo_h5_list.append(file[:-3])

for file1 in geom_inter:
	dat_h5 = [x for x in dat_h5_list if file1 in x][0]
	geom_h5 = [x for x in geo_h5_list if file1 in x][0]
	geom_hf = h5py.File(geom_h5,'r')
	lat = geom_hf['All_Data']['VIIRS-IMG-GEO-TC_All']['Latitude'][:]
	ys, xs = lat.shape
	h5toGTiff(dat_h5, geom_h5, xs, ys, '/share/ssd-scratch/htranvie/Flood/data/SVI01_%s_temp.tif' %file1)



