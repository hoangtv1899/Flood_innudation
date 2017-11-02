#!/data/apps/anaconda/2.7-4.3.1/bin/python

import os
from glob import glob
from datetime import datetime
import pandas as pd

#block writing functions
def writeBasic(dem='basic\dem.tif',
				ddm='basic\\fdr.tif',
				fac='basic\\fac.tif',proj='geographic',esriddm='false',selffam='true'):
	block ='[Basic]\n'
	block +='DEM='+dem+'\n'
	block +='DDM='+ddm+'\n'
	block +='FAM='+fac+'\n'
	block +='PROJ='+proj+'\n'
	block +='ESRIDDM='+esriddm+'\n'
	block +='selfFAM='+selffam+'\n'
	return block

def writePrecip(header='CCS',type='TIF',unit='mm/d',freq='d',loc='precip\\',
				name='CCS_1dYYYYMMDD.tif'):
	block = '[PrecipForcing '+header+']\n'
	block += 'TYPE='+type+'\n'
	block += 'UNIT='+unit+'\n'
	block += 'FREQ='+freq+'\n'
	block += 'LOC='+loc+'\n'
	block += 'NAME='+name+'\n'
	return block

def writePET(header='FEWSNET',type='TIF',unit='mm/d',freq='d',loc='pet\\',
				name='etYYYYMMDD.bil.tif'):
	block = '[PETForcing '+header+']\n'
	block += 'TYPE='+type+'\n'
	block += 'UNIT='+unit+'\n'
	block += 'FREQ='+freq+'\n'
	block += 'LOC='+loc+'\n'
	block += 'NAME='+name+'\n'
	return block

def writeGauge(dict_gauge,output='true'):
	block = '[Gauge '+str(dict_gauge['site_no'])+']\n'
	block += 'LON='+str(dict_gauge['dec_long_va'])+'\n'
	block += 'LAT='+str(dict_gauge['dec_lat_va'])+'\n'
	block += 'OBS='+str(dict_gauge['obs'])+'\n'
	if np.isnan(dict_gauge['basin_area']) == False:
		block += 'BASINAREA='+str(dict_gauge['basin_area'])+'\n'
	block += 'OUTPUTTS='+output+'\n'
	return block

def writeBasin(header,list_dict_gauges):
	block = '[Basin '+header+']\n'
	for gauge in list_dict_gauges:
		block += 'GAUGE='+str(gauge['site_no'])+'\n'
	return block

def writeCrestParamSet(header,list_dict_gauges):
	block = '[CrestParamSet '+header+']\n'
	for dict_gauge in list_dict_gauges:
		block += 'gauge='+str(dict_gauge['site_no'])+'\n'
		block += 'wm='+str(dict_gauge['wm'])+'\n'
		block += 'b='+str(dict_gauge['b'])+'\n'
		block += 'im='+str(dict_gauge['im'])+'\n'
		block += 'ke='+str(dict_gauge['ke'])+'\n'
		block += 'fc='+str(dict_gauge['fc'])+'\n'
		block += 'iwu='+str(dict_gauge['iwu'])+'\n'
	return block

def writeKwparamset(header,list_dict_gauges1):
	block = '[kwparamset '+header+']\n'
	for dict_gauge in list_dict_gauges1:
		block += 'gauge='+str(dict_gauge['site_no'])+'\n'
		block += 'under='+str(dict_gauge['under'])+'\n'
		block += 'leaki='+str(dict_gauge['leaki'])+'\n'
		block += 'alpha='+str(dict_gauge['alpha'])+'\n'
		block += 'beta='+str(dict_gauge['beta'])+'\n'
		block += 'alpha0='+str(dict_gauge['alpha0'])+'\n'
		block += 'th='+str(dict_gauge['th'])+'\n'
		block += 'isu='+str(dict_gauge['isu'])+'\n'
	return block

def writeSimpleinundationparamset(header,list_dict_gauges2):
	block = '[simpleinundationparamset '+header+']\n'
	for dict_gauge in list_dict_gauges2:
		block += 'gauge='+str(dict_gauge['site_no'])+'\n'
		block += 'alpha='+str(dict_gauge['alpha_inun'])+'\n'
		block += 'beta='+str(dict_gauge['beta_inun'])+'\n'
	return block

def writeCrestCaliParams(header,name,obj='nsce',dream_ndraw=100):
	block = '[CrestCaliParams '+header+']\n'
	block += 'gauge='+name+'\n'
	block += 'objective='+obj+'\n'
	block += 'dream_ndraw='+str(dream_ndraw)+'\n'
	block += 'wm=5.0,250.0\n'
	block += 'b=0.1,20.0\n'
	block += 'im=0.009999,0.5\n'
	block += 'ke=0.001,1.0\n'
	block += 'fc=0.0,150.0\n'
	block += 'iwu=24.999,25.0\n'
	return block

def writeKwcaliparams(header,name):
	block = '[kwcaliparams '+header+']\n'
	block += 'gauge='+name+'\n'
	block += 'alpha=0.01,3.0\n'
	block += 'alpha0=0.01,5.0\n'
	block += 'beta=0.01,1.0\n'
	block += 'under=0.0001,3.0\n'
	block += 'leaki=0.01,1.0\n'
	block += 'th=1.0,10.0\n'
	block += 'isu=0.0,0.00001\n'
	return block

def writeTask(header,style,basin,precip,pet,param_set,routing_set,inundation_set,
				timeStep,timeStart,timeWarm,timeEnd,
				cali_param=None,routing_cali=None,
				model='CREST',routing='KW',inundation='SIMPLEINUNDATION',
				output='output\\',outputGrid='INUNDATION'):
	block = '[Task '+header+']\n'
	block += 'MODEL='+model+'\n'
	block += 'ROUTING='+routing+'\n'
	block += 'BASIN='+basin+'\n'
	block += 'PRECIP='+precip+'\n'
	block += 'PET='+pet+'\n'
	block += 'INUNDATION='+inundation+'\n'
	block += 'OUTPUT='+output+'\n'
	block += 'PARAM_SET='+param_set+'\n'
	block += 'ROUTING_PARAM_Set='+routing_set+'\n'
	block += 'INUNDATION_PARAM_SET='+inundation_set+'\n'
	block += 'STYLE='+style+'\n'
	if style == 'cali_dream':
		block += 'cali_param='+cali_param+'\n'
		block += 'routing_cali_param='+routing_cali+'\n'
		#block += 'OUTPUT_GRIDS='+outputGrid+'\n'
	block += 'TIMESTEP='+timeStep+'\n'
	block += 'TIME_BEGIN='+timeStart+'\n'
	block += 'TIME_WARMEND='+timeWarm+'\n'
	block += 'TIME_END='+timeEnd+'\n'
	block += 'OUTPUT_GRIDS='+outputGrid+'\n'
	return block
	
def writeExecute(task):
	block='[Execute]\n'
	block+='TASK='+task+'\n'
	return block


#load station file
sites = pd.read_csv('final_site.csv')
#load list gauge files
gauge_files = sorted(glob('gauge/*.csv'))
gauge_files = ['obs\\'+os.path.basename(x) for x in gauge_files]
#modify the data frame
#crest params set
sites['obs'] = gauge_files
sites['wm'] = [50]*len(gauge_files)
sites['b'] = [0.2]*len(gauge_files)
sites['im'] = [0.154]*len(gauge_files)
sites['ke'] = [0.4]*len(gauge_files)
sites['fc'] = [50]*len(gauge_files)
sites['iwu'] = [20]*len(gauge_files)
#kw params set
sites['under'] = [3]*len(gauge_files)
sites['leaki'] = [0.042]*len(gauge_files)
sites['alpha'] = [1.5]*len(gauge_files)
sites['beta'] = [0.5]*len(gauge_files)
sites['alpha0'] = [1.174]*len(gauge_files)
sites['th'] = [4.031]*len(gauge_files)
sites['isu'] = [0]*len(gauge_files)
#inundation params set
sites['alpha_inun'] = [1.5]*len(gauge_files)
sites['beta_inun'] = [0.5]*len(gauge_files)

list_sites = sites.T.to_dict().values()

#start writing 
with open('control.txt','w') as fo:
	line1 = writeBasic()
	fo.write(line1)
	line2 = writePrecip()
	fo.write(line2)
	line3 = writePET()
	fo.write(line3)
	for site in list_sites:
		line4 = writeGauge(site)
		fo.write(line4)
	line5 = writeBasin('Nebraska',list_sites)
	fo.write(line5)
	line6 = writeCrestParamSet('Neb_Crest_params',list_sites)
	fo.write(line6)
	line7 = writeKwparamset('Neb_kw_params',list_sites)
	fo.write(line7)
	line8 = writeSimpleinundationparamset('Neb_inun_params',list_sites)
	fo.write(line8)
	line9 = writeCrestCaliParams('Neb_Crest_cali1',str(list_sites[0]['site_no']))
	fo.write(line9)
	line10 = writeKwcaliparams('Neb_kw_cali1',str(list_sites[0]['site_no']))
	fo.write(line10)
	line11 = writeTask('Neb_task','cali_dream','Nebraska','CCS','FEWSNET',
						'Neb_Crest_params','Neb_kw_params','Neb_inun_params',
						'd','201301010000','201302010000','201312310000',
						'Neb_Crest_cali1','Neb_kw_cali1')
	fo.write(line11)
	line12 = writeExecute('Neb_task')
	fo.write(line12)




