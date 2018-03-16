#-------------------------------------------------------------------------------
# Name:        SCE_Python_shared version
# This is the implementation for the SCE algorithm,
# written by Q.Duan, 9/2004 - converted to python by Van Hoey S.2011
# Purpose:
# Dependencies: 	numpy
#
# Author:      VHOEYS
#
# Created:     11/10/2011
# Copyright:   (c) VHOEYS 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

## Refer to paper:
##  'EFFECTIVE AND EFFICIENT GLOBAL OPTIMIZATION FOR CONCEPTUAL
##  RAINFALL-RUNOFF MODELS' BY DUAN, Q., S. SOROOSHIAN, AND V.K. GUPTA,
##  WATER RESOURCES RESEARCH, VOL 28(4), PP.1015-1031, 1992.
##

import random
import numpy as np
import time
import ctypes
import multiprocessing as mp
from multiprocessing import Manager
from contextlib import closing

from SCE_functioncall import *

def cceua(s,sf,bl,bu,iseed):
#  This is the subroutine for generating a new point in a simplex
#
#   s(.,.) = the sorted simplex in order of increasing function values
#   s(.) = function values in increasing order
#
# LIST OF LOCAL VARIABLES
#   sb(.) = the best point of the simplex
#   sw(.) = the worst point of the simplex
#   w2(.) = the second worst point of the simplex
#   fw = function value of the worst point
#   ce(.) = the centroid of the simplex excluding wo
#   snew(.) = new point generated from the simplex
#   iviol = flag indicating if constraints are violated
#         = 1 , yes
#         = 0 , no
	nps,nopt=s.shape
	n = nps
	m = nopt
	alpha = 1.0
	beta = 0.5
	# Assign the best and worst points:
	sb=s[0,:]
	fb=sf[0]
	sw=s[-1,:]
	fw=sf[-1]
	# Compute the centroid of the simplex excluding the worst point:
	ce= np.mean(s[:-1,:],axis=0)
	# Attempt a reflection point
	snew = ce + alpha*(ce-sw)
	# Check if is outside the bounds:
	ibound=0
	s1=snew-bl
	idx=(s1<0).nonzero()
	if idx[0].size <> 0:
		ibound=1
	s1=bu-snew
	idx=(s1<0).nonzero()
	if idx[0].size <> 0:
		ibound=2
	if ibound >= 1:
		snew = SampleInputMatrix(1,nopt,bu,bl,iseed,distname='randomUniform')[0]  #checken!!
	##    fnew = functn(nopt,snew);
	fnew = EvalObjF(nopt,snew)
	# Reflection failed; now attempt a contraction point:
	if fnew > fw:
		snew = sw + beta*(ce-sw)
		fnew = EvalObjF(nopt,snew)
	# Both reflection and contraction have failed, attempt a random point;
		if fnew > fw:
			snew = SampleInputMatrix(1,nopt,bu,bl,iseed,distname='randomUniform')[0]  #checken!!
			fnew = EvalObjF(nopt,snew)
	# END OF CCE
	return snew,fnew

#mini functions for parallel running

def init0(nopt_,x_flatten_,xf_):
	global nopt,x_flatten,xf
	nopt = nopt_
	x_flatten = x_flatten_
	xf = xf_

def init1(nopt_,cx_,cf_,iseed_,bu_,bl_):
	global nopt,cx,cf,iseed,bu,bl
	nopt = nopt_
	cx = cx_
	cf = cf_
	iseed = iseed_
	bu = bu_
	bl = bl_

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

def EvalParallel(i):
	arr1 = tonumpyarray(xf)
	xi = tonumpyarray(x_flatten)[nopt*i:nopt*(i+1)]
	results = EvalObjF(nopt,xi)
	arr1[i] = results

def EvolveParallel(loop):
	npg = 2*nopt+1
	nps = nopt+1
	cx_par = tonumpyarray(cx).reshape(npg,nopt)
	cf_par = tonumpyarray(cf)
	# Select simplex by sampling the complex according to a linear
	# probability distribution
	lcs=np.array([0]*nps)
	lcs[0] = 1
	for k3 in range(1,nps):
		for i in range(1000):
##                        lpos = 1 + int(np.floor(npg+0.5-np.sqrt((npg+0.5)**2 - npg*(npg+1)*random.random())))
			lpos = int(np.floor(npg+0.5-np.sqrt((npg+0.5)**2 - npg*(npg+1)*random.random())))
##                        idx=find(lcs(1:k3-1)==lpos)
			idx=(lcs[0:k3]==lpos).nonzero()  #check of element al eens gekozen
			if idx[0].size == 0:
				break
			
		lcs[k3] = lpos
	lcs.sort()
	
	# Construct the simplex:
	s = np.zeros((nps,nopt))
	s=cx_par[lcs,:]
	sf = cf_par[lcs]
	
	snew,fnew=cceua(s,sf,bl,bu,iseed)
	
	# Replace the worst point in Simplex with the new point:
	s[-1,:] = snew
	sf[-1] = fnew
	
	# Replace the simplex into the complex;
	cx_par[lcs,:] = s
	cf_par[lcs] = sf
	
	# Sort the complex;
	idx = np.argsort(cf_par)
	cf_par = np.sort(cf_par)
	cx_par=cx_par[idx,:]

def ComplexLoop(igs,nopt,ngs,iseed,x_arr,xf_arr,bu,bl):
	# Partition the population into complexes (sub-populations);
	npg = 2*nopt+1
	nspl = npg
	nps = nopt+1
	cx=mp.Array(ctypes.c_long,npg*nopt)
	cf=mp.Array(ctypes.c_long,npg)
	k1=np.array(range(npg))
	k2=k1*ngs+igs
	cx_arr = tonumpyarray(cx).reshape(npg,nopt)
	cf_arr = tonumpyarray(cf)
	cx_arr[k1,:] = x_arr[k2,:]
	cf_arr[k1] = xf_arr[k2]
	
	# Evolve sub-population igs for nspl steps:
	start2 = time.time()
	with closing(mp.Pool(initializer=init1, initargs=(nopt,cx,cf,iseed,bu,bl,))) as p1:
		p1.map_async(EvolveParallel,range(nspl))
	p1.join()
	end2 = time.time()
	print 'time elapsed for evolve parallel: '+str(end2-start2)
	# End of Inner Loop for Competitive Evolution of Simplexes
	#end of Evolve sub-population igs for nspl steps:
	
	# Replace the complex back into the population;
	x_arr[k2,:] = cx_arr[k1,:]
	xf_arr[k2] = cf_arr[k1]
	# End of Loop on Complex Evolution;
	return x_arr, xf_arr
	

def sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg):
# This is the subroutine implementing the SCE algorithm,
# written by Q.Duan, 9/2004 - converted to python by Van Hoey S.2011
#
# Definition:
#  x0 = the initial parameter array at the start; np.array
#     = the optimized parameter array at the end;
#  f0 = the objective function value corresponding to the initial parameters
#     = the objective function value corresponding to the optimized parameters
#  bl = the lower bound of the parameters; np.array
#  bu = the upper bound of the parameters; np.array
#  iseed = the random seed number (for repetetive testing purpose)
#  iniflg = flag for initial parameter array (=1, included it in initial
#           population; otherwise, not included)
#  ngs = number of complexes (sub-populations)
#  npg = number of members in a complex
#  nps = number of members in a simplex
#  nspl = number of evolution steps for each complex before shuffling
#  mings = minimum number of complexes required during the optimization process
#  maxn = maximum number of function evaluations allowed during optimization
#  kstop = maximum number of evolution loops before convergency
#  percento = the percentage change allowed in kstop loops before convergency
	
# LIST OF LOCAL VARIABLES
#    x(.,.) = coordinates of points in the population
#    xf(.) = function values of x(.,.)
#    xx(.) = coordinates of a single point in x
#    cx(.,.) = coordinates of points in a complex
#    cf(.) = function values of cx(.,.)
#    s(.,.) = coordinates of points in the current simplex
#    sf(.) = function values of s(.,.)
#    bestx(.) = best point at current shuffling loop
#    bestf = function value of bestx(.)
#    worstx(.) = worst point at current shuffling loop
#    worstf = function value of worstx(.)
#    xnstd(.) = standard deviation of parameters in the population
#    gnrng = normalized geometric mean of parameter ranges
#    lcs(.) = indices locating position of s(.,.) in x(.,.)
#    bound(.) = bound on ith variable being optimized
#    ngs1 = number of complexes in current population
#    ngs2 = number of complexes in last population
#    iseed1 = current random seed
#    criter(.) = vector containing the best criterion values of the last
#                10 shuffling loops
	
    # Initialize SCE parameters:
	nopt=x0.size
	npg=2*nopt+1
	nps=nopt+1
	nspl=npg
	mings=ngs
	npt=npg*ngs

	bound = bu-bl  #np.array

	# Create an initial population to fill array x(npt,nopt):
	x = SampleInputMatrix(npt,nopt,bu,bl,iseed,distname='randomUniform')
	if iniflg==1:
		x[0,:]=x0

	nloop=0
	icall=0
	#Run parallel to get cost function
	xf=mp.Array(ctypes.c_long,npt)
	x_flatten = mp.Array(ctypes.c_long,npt*nopt)
	x_arr = tonumpyarray(x_flatten).reshape(npt,nopt)
	x_arr[:] = x.copy()
	start1 = time.time()
	with closing(mp.Pool(initializer=init0, initargs=(nopt,x_flatten,xf,))) as p:
		p.map_async(EvalParallel,range(npt))
	p.join()
	end1 = time.time()
	print 'Total time to get 1st time cost function '+str(end1-start1)+' seconds'
	#Finish get cost function
	xf_arr = tonumpyarray(xf)
	xf_arr[np.isnan(xf_arr)] = 1.

	#f0=xf[0]

	# Sort the population in order of increasing function values;
	idx = np.argsort(xf_arr)
	xf_arr = np.sort(xf_arr)
	x_arr=x_arr[idx,:]

	# Record the best and worst points;
	bestx=x_arr[0,:]
	bestf=xf_arr[0]
	worstx=x_arr[-1,:]
	worstf=xf_arr[-1]

	BESTF=bestf
	BESTX=bestx
	ICALL=icall

	# Compute the standard deviation for each parameter
	xnstd=np.std(x_arr,axis=0)

	# Computes the normalized geometric range of the parameters
	gnrng=np.exp(np.mean(np.log((np.max(x_arr,axis=0)-np.min(x_arr,axis=0))/bound)))

	print 'The Initial Loop: 0'
	print ' BESTF:  %f ' %bestf
	print ' BESTX:  '
	print bestx
	print ' WORSTF:  %f ' %worstf
	print ' WORSTX: '
	print worstx
	print '     '

	# Check for convergency;
	if icall >= maxn:
		print '*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT'
		print 'ON THE MAXIMUM NUMBER OF TRIALS '
		print maxn
		print 'HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:'
		print icall
		print 'OF THE INITIAL LOOP!'

	if gnrng < peps:
		print 'THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE'

	# Begin evolution loops:
	nloop = 0
	criter=[]
	criter_change=1e+5

	while icall<maxn and gnrng>peps and criter_change>pcento:
		nloop+=1
		
		# Loop on complexes (sub-populations);
		for igs in range(ngs):
			x_arr, xf_arr = ComplexLoop(igs,nopt,ngs,iseed,x_arr,xf_arr,bu,bl)
		
		icall += ngs**2
		# End of Loop on Complex Evolution;
		# Shuffled the complexes;
		xf_arr[np.isnan(xf_arr)] = 1.
		idx = np.argsort(xf_arr)
		xf_arr = np.sort(xf_arr)
		x_arr=x_arr[idx,:]
		
		#PX=x
		#PF=xf
		
		# Record the best and worst points;
		bestx=x_arr[0,:]
		bestf=xf_arr[0]
		worstx=x_arr[-1,:]
		worstf=xf_arr[-1]
		
		BESTX = np.append(BESTX,bestx, axis=0) #appenden en op einde reshapen!!
		BESTF = np.append(BESTF,bestf)
		ICALL = np.append(ICALL,icall)
		
		# Compute the standard deviation for each parameter
		xnstd=np.std(x_arr,axis=0)
		
		# Computes the normalized geometric range of the parameters
		gnrng=np.exp(np.mean(np.log((np.max(x_arr,axis=0)-np.min(x_arr,axis=0))/bound)))
		
		print 'Evolution Loop: %d  - Trial - %d' %(nloop,icall)
		print ' BESTF:  %f ' %bestf
		print ' BESTX:  '
		print bestx
		print ' WORSTF:  %f ' %worstf
		print ' WORSTX: '
		print worstx
		print '     '
		
		# Check for convergency;
		if icall >= maxn:
			print '*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT'
			print 'ON THE MAXIMUM NUMBER OF TRIALS '
			print maxn
			print 'HAS BEEN EXCEEDED.'
		
		if gnrng < peps:
			print 'THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE'
		
		criter=np.append(criter,bestf)
		
		if nloop >= kstop: #nodig zodat minimum zoveel doorlopen worden
			criter_change= np.abs(criter[nloop-1]-criter[nloop-kstop])*100
			criter_change= criter_change/np.mean(np.abs(criter[nloop-kstop:nloop]))
			if criter_change < pcento:
				print 'THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY LESS THAN THE THRESHOLD %f' %(kstop,pcento)
				print 'CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!'

	# End of the Outer Loops
	print 'SEARCH WAS STOPPED AT TRIAL NUMBER: %d' %icall
	print 'NORMALIZED GEOMETRIC RANGE = %f'  %gnrng
	print 'THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY %f' %(kstop,criter_change)

	#reshape BESTX
	BESTX=BESTX.reshape(BESTX.size/nopt,nopt)

	# END of Subroutine sceua
	return bestx,bestf,BESTX,BESTF,ICALL