#!/data/apps/enthought_python/2.7.3/bin/python

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
from time import time

def animate(i):
	arr0 = res_arr[i,:,:]
	im1.set_array(np.flipud(arr0))
	ytext.set_text((t0 + timedelta(days=i)).strftime('%Y-%m-%d'))
	return im1, ytext

#load data
list_res = sorted(glob('results1/*.wd'))
res_arr = np.zeros((len(list_res),201,124))
for i,res_f in enumerate(list_res):
	temp_arr = np.loadtxt(res_f,skiprows=6)
	res_arr[i,:,:] = temp_arr

t0 = datetime(2017,4,1)
fig =plt.figure()
ax = fig.add_subplot(111)
m1 = Basemap(llcrnrlon=-90.39,llcrnrlat=37.3,urcrnrlon=-89.39,urcrnrlat=38.63,\
				projection='mill',lon_0=0)
m1.drawparallels(np.arange(-35.,40.,0.5),labels=[1,0,0,0], linewidth=0.05)
m1.drawmeridians(np.arange(-90.,-80.,0.5),labels=[0,0,0,1], linewidth=0.05)
im1=m1.imshow(np.flipud(res_arr[0,:,:]),cmap='Blues')
m1.colorbar(location='right',pad='5%')
#plt.clim(0,15)
ytext = ax.text(0.65,0.95,(t0 + timedelta(days=0)).strftime('%Y-%m-%d'),transform=ax.transAxes)

anim = animation.FuncAnimation(fig, animate,
                               frames=61, interval=31)
anim.save('river_animation1.gif', writer='imagemagick', fps=5)
