#!/data/apps/anaconda/2.7-4.3.1/bin/python

import pandas as pd
import numpy as np
from glob import glob

list_f = glob('output/*.csv')
res = pd.DataFrame(columns=['station','coef','bias','nsce','rmse'])
for i,f in enumerate(list_f):
	df = pd.read_csv(f)
	station = f.split('ts.')[1].split('.crest')[0]
	dis = df['Discharge(m^3 s^-1)']
	obs = df['Observed(m^3 s^-1)']
	coeff = np.corrcoef(obs,dis)[0,1]
	bias = 100*(np.sum(dis)-np.sum(obs))/np.sum(obs)
	obs_mean = np.ones((len(obs),))*np.mean(obs)
	diff_dis = obs - dis
	diff_avg = obs - obs_mean
	nsce = 1 - np.sum(np.square(diff_dis))/np.sum(np.square(diff_avg))
	rmse = np.sqrt(np.mean((obs-dis)**2))
	res.loc[i] = [station,coeff,bias,nsce,rmse]

res.to_csv('results.csv',index=False)