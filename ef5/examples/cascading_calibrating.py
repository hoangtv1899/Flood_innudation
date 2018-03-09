#!/data/apps/anaconda/2.7-4.3.1/bin/python

import os
from glob import glob
from datetime import datetime
import pandas as pd

def ParseGaugeInfo(content,i):
	res = {}
	next_line = content[i+1].strip()
	while next_line and (next_line[0]!='[' and next_line[-1]!=']'):
		info = next_line.split('=')
		res[info[0]] = info[1]
		i += 1
		if i+1 >= len(content):
			return res
		next_line = content[i+1].strip()
	return res

def ParseBasinInfo(content,i):
	res = []
	next_line = content[i+1].strip()
	while next_line and (next_line[0]!='[' and next_line[-1]!=']'):
		info = next_line.split('=')
		res.append(info[1])
		i += 1
		next_line = content[i+1].strip()
	return res

def ParseLoopInfo(content,i):
	res = {}
	next_line = content[i+1].strip()
	while next_line and (next_line[0]!='[' and next_line[-1]!=']'):
		info = next_line.split('=')
		if info[0] in res:
			res[info[0]].append(info[1])
		else:
			res[info[0]] = [info[1]]
		i += 1
		next_line = content[i+1].strip()
	return res

def ReadCT(content):
	stationMeta = []
	calibratingGauge = {}
	curr_style = ''
	for i,line in enumerate(content):
		line = line.strip()
		if (line and line[0]=='[') and (line[-1]==']'):
			headers = line[1:-1].split(' ')
			if headers[0] == 'Gauge':
				res = ParseGaugeInfo(content,i)
				if headers[1] not in [x['Name'] for x in stationMeta]:
					res['Name'] = headers[1]
					stationMeta.append(res)
			elif headers[0] == 'Basin':
				if set(ParseBasinInfo(content,i)) != set([x['Name'] for x in stationMeta]):
					print 'please check all the registered gauges'
					break
			elif headers[0] == 'CrestParamSet':
				res1 = ParseLoopInfo(content,i)
			elif headers[0] == 'kwparamset':
				res1.update(ParseLoopInfo(content,i))
			elif headers[0] == 'simpleinundationparamset':
				res2 = ParseLoopInfo(content,i)
				res2['alpha_inun'] = res2.pop('alpha')
				res2['beta_inun'] = res2.pop('beta')
				res1.update(res2)
				for j, name in enumerate(res1['gauge']):
					for idx, item in enumerate(stationMeta):
						if item['Name'] == name:
							for k in res1.keys():
								item[k] = res1[k][j]
							stationMeta[idx] = item
							break
			elif headers[0] == 'CrestCaliParams':
				res3 = ParseGaugeInfo(content,i)
				calibratingGauge['Name'] = res3['gauge']
				calibratingGauge['ndraw'] = res3['dream_ndraw']
				calibratingGauge['lines'] = [i+1]
			elif headers[0] == 'kwcaliparams':
				calibratingGauge['lines'].append(i+1)
			elif headers[0] == 'Task':
				curr_style = ParseGaugeInfo(content,i)['STYLE']
	
	return pd.DataFrame(stationMeta), calibratingGauge,curr_style

CTfile = 'control.txt'
#Open control file
content = open(CTfile,'r').readlines()

#loop through to find:
#1. gauges
#2. their parameters
#3. which one is calibrating
#4. time period

stationMeta, calibratingGauge,curr_style = ReadCT(content)
#os.system('ef5 > status.txt')
#stationRerun = ['Brainerd','Brooklyn_park','Henderson','Snelling','Wisconsin_dells','Joslin']
for jk,station in enumerate(stationMeta.gauge.tolist()):
	os.system('ef5')
	CTfile = 'control.txt'
	#Open control file
	content = open(CTfile,'r').readlines()
	stationMeta1, calibratingGauge,curr_style = ReadCT(content)
	res_file = glob('output/cali_dream.'+calibratingGauge['Name'].lower()+'*.csv')[0]
	res_content = open(res_file,'r').readlines()
	for i, line in enumerate(res_content):
		line = line.strip()
		if '[WaterBalance]' in line:
			res1 = ParseGaugeInfo(res_content,i)
		elif '[Routing]' in line:
			res1.update(ParseGaugeInfo(res_content,i))
	
	res1['gauge'] = calibratingGauge['Name']
	
	with open('control.txt','w') as fo:
		idxs = [idx for idx, x in enumerate(content) if 'gauge='+calibratingGauge['Name'] in x]
		idxs = range(idxs[0]+1,idxs[0]+7)+range(idxs[1]+1,idxs[1]+8)
		for i,line in enumerate(content):
			if i in idxs:
				line0 = line.strip()
				vals = line0.split('=')
				line = line.replace(vals[1],res1[vals[0]])
			if i in calibratingGauge['lines']:
				if jk < len(stationMeta.gauge.tolist()) - 1:
					line = line.replace(calibratingGauge['Name'],stationMeta.gauge.tolist()[jk+1])
			if 'STYLE' in line:
				line = 'STYLE=cali_dream\n'
			fo.write(line)
	





