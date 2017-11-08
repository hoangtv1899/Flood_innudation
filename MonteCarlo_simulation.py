#!/data/apps/anaconda/2.7-4.3.1/bin/python

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
import os
import numpy as np
from glob import glob

winonaBasin = io.imread('elevation/winona_basin.tif')
CCimgs = sorted(glob('Cloud_free/*.npy'))
CCtimeseries = np.array([],dtype=np.int16).reshape(0,winonaBasin.shape[0],winonaBasin.shape[1])
for img in CCimgs:
	arr = np.load(img)
	CCtimeseries = np.vstack([arr.reshape(-1,winonaBasin.shape[0],winonaBasin.shape[1]),
								CCtimeseries])

totalInunPixel = np.sum(CCtimeseries[:,winonaBasin==3]>1,axis=1)
totalPixel = np.sum(CCtimeseries[:,winonaBasin==3]>0,axis=1)
f = totalInunPixel/totalPixel.astype(np.float)

f[f>0.5] = f[f>0.5]+0.45
f[np.logical_and(f<=0.5,f>0.3)] = f[np.logical_and(f<=0.5,f>0.3)]+0.3
f[np.logical_and(f<=0.3,f>0.001)] = f[np.logical_and(f<=0.3,f>0.001)]+0.1