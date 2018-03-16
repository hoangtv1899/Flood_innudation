#-------------------------------------------------------------------------------
# Name:        Testfile for Shuffled Complex Evolution Algorithm implementation
# Purpose:      Call one of the example functions to run and find the optimum
#               Visualize the 2D (or 3D) objective function +trace of the BESTX
#
#               if matplotlib is no available, comment the plot part
#
# Author:      VHOEYS
#
# Created:     11/10/2011
# Copyright:   (c) VHOEYS 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import os
import sys
import time
import numpy as np

#Call the scripts with SCE and
from SCE_python import *
from SCE_functioncall import *

################################################################################
# PARAMETERS TO TUNE THE ALGORITHM
# Definition:
#  iseed = the random seed number (for repetetive testing purpose;pos integers)
#  iniflg = flag for initial parameter array (=1, included it in initial
#           population; otherwise, not included)
#  ngs = number of complexes (sub-populations)
# peps = value of NORMALIZED GEOMETRIC RANGE needed for convergence
#  maxn = maximum number of function evaluations allowed during optimization
#  kstop = maximum number of evolution loops before convergency
#  pcento = the percentage change allowed in kstop loops before convergency
#-------------------------------------------------------------------------------
maxn=10000
kstop=30
pcento=0.001
peps=0.001
iseed= 0
iniflg=0
ngs=4
################################################################################


################################################################################
# PARAMETERS FOR OPTIMIZATION PROBLEM
# Definition:
#  x0 = the initial parameter array at the start; np.array
#     = the optimized parameter array at the end;
#  f0 = the objective function value corresponding to the initial parameters
#     = the objective function value corresponding to the optimized parameters
#  bl = the lower bound of the parameters; np.array
#  bu = the upper bound of the parameters; np.array
################################################################################
start = time.clock()
bu=np.hstack([np.ones(12)*4,np.ones(12)])
bl=np.zeros(24)
x0=np.zeros(24)
bestx,bestf,BESTX,BESTF,ICALL = sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg)

elapsed = (time.clock() - start)
print 'The calculation of the SCE algorithm took %f seconds' %elapsed

################################################################################
##  PLOT PART
################################################################################

'''
plot the trace of the parametersvalue
'''
import matplotlib.pyplot as plt
fig=plt.figure()
ax1=plt.subplot(121)
ax1.plot(BESTX)
plt.title('Trace of the different parameters')
plt.xlabel('Evolution Loop')
plt.ylabel('Parvalue')

'''
Plot the parmaeterspace in 2D with trace of the BESTX
'''
#- - - - - - - - - - - - - - - - - - - - - - - - -
#   make these smaller to increase the resolution
dx, dy = 0.05, 0.05
if foo == 5:
    print 'Use big enough steps to visualise parameter space'
##example 5 needs bigger steps!!!
##dx, dy = 5., 5.
#- - - - - - - - - - - - - - - - - - - - - - - - -

ax2=plt.subplot(122)
x = np.arange(bl[0], bu[0], dx)
y = np.arange(bl[1], bu[1], dy)
X,Y = np.meshgrid(x, y)

parspace=np.zeros((x.size,y.size))
for i in range(x.size):
    x1=x[i]
    for j in range(y.size):
        x2=y[j]
        parspace[i,j]=EvalObjF(x0.size,np.array([x1,x2]),testcase=True,testnr=foo)

ax2.pcolor(X, Y, parspace)
##plt.colorbar()
ax2.plot(BESTX[:,0],BESTX[:,1],'*')
plt.title('Trace of the BESTX parameter combinations')
plt.xlabel('PAR 1')
plt.ylabel('PAR 2')

'''
Plot the parmaeterspace in 3D - commented out
'''
##from mpl_toolkits.mplot3d import Axes3D
##from matplotlib import cm
##from matplotlib.ticker import LinearLocator, FormatStrFormatter
##fig=plt.figure()
##ax = fig.gca(projection='3d')
##surf = ax.plot_surface(X, Y, parspace, rstride=8, cstride=8, cmap=cm.jet,linewidth=0, antialiased=False)
####cset = ax.contourf(X, Y, parspace, zdir='z', offset=-100)
##fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()