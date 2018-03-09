#-------------------------------------------------------------------------------
# Name:        SCE_Python_shared version -  helpfunctions
# This is the implementation for the SCE algorithm,
# written by Q.Duan, 9/2004 - converted to python by Van Hoey S. 2011
# Purpose:
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
import numpy as np
from glob import glob
import gdal
import time
import pandas as pd
from subprocess import Popen, PIPE, STDOUT
import inspect
import shutil

##print sys.path[0]

##sys.path.append('D:\Modellen\Snippets_for_all\SCE')
################################################################################
## Modrun  is the user part of the script:
## gets number of parameters and an array x with these parameters
## let your function or model run/evaluate and give the evaluation value back
##------------------------------------------------------------------------------

def createTif(fname, res_arr, geom, dtype=gdal.GDT_Int16, ndata=-99):
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(
			fname,
			res_arr.shape[1],
			res_arr.shape[0],
			1,
			dtype,
			options = ['COMPRESS=LZW'])
	dataset.SetGeoTransform(geom)
	outband = dataset.GetRasterBand(1)
	outband.WriteArray(res_arr)
	outband.FlushCache()
	outband.SetNoDataValue(ndata)
	dataset.FlushCache()

def catValidation(ref_arr, mod_arr):
	val = []
	type = []
	#hit
	hit_arr = np.logical_and(mod_arr==1,ref_arr==1)
	no_hit = np.sum(hit_arr)
	#miss
	miss_arr = np.logical_and(mod_arr==0,ref_arr==1)
	no_miss = np.sum(miss_arr)
	#false
	false_arr = np.logical_and(mod_arr==1,ref_arr==0)
	no_false = np.sum(false_arr)
	#correct negative
	corrneg = np.logical_and(mod_arr==0,ref_arr==0)
	no_corrneg = np.sum(corrneg)
	val += [no_hit,no_miss,no_false,no_corrneg]
	type += ['Hit','Miss','False','Correct Negative']
	val += [no_hit/(float(no_miss)+no_hit), no_false/(float(no_hit)+no_false), (no_hit*no_corrneg-no_false*no_miss)/float((no_hit+no_miss)*(no_corrneg+no_false))]
	type += ['POD','FAR','HK']
	return val, type

def Validation(d0,arr_cf,arr_wd):
	results = pd.DataFrame()
	arr_cf_bin = (arr_cf>0).astype(np.int)
	arr_wd_bin = (arr_wd>0.8).astype(np.int)
	val_mod, type_mod = catValidation(arr_cf_bin, arr_wd_bin)
	results['Date'] = [d0]*7
	results['Types'] = type_mod
	results['Values'] = val_mod
	return results

def ResampleImage(img,img1,resolution,bounding_box,srcnodata=-99):
	os.system('gdalwarp -overwrite -srcnodata '+str(srcnodata)+' -dstnodata -99 -tr '+\
				str(resolution)+' '+str(resolution)+\
				' -te '+' '.join([str(x) for x in bounding_box])+' '+img+' '+img1)

def Modrun(npar,x):
	#working directory
	wdir = '/pub/htranvie/second_year/hoang_thesis/Flood/'
	mdir = wdir+'ef5/mississippi_flood_map/'
	#Read alpha and beta from x
	alpha, beta = x[:len(x)/2], x[len(x)/2:]
	#Open control file
	content = open(mdir+'control1.txt','r').readlines()
	#Modify control file based on new parameters
	for k,line in enumerate(content):
		line = line.strip()
		if (line and line[0]=='[') and (line[-1]==']'):
			headers = line[1:-1].split(' ')
			if headers[0] == 'simpleinundationparamset':
				count = 0
				next_line = content[k+1+count].strip()
				while (next_line and next_line[0]!='[') and (next_line[-1]!=']'):
					try:
						info = next_line.split('=')
						if info[0]=='alpha':
							next_line = next_line.replace(info[1],str(alpha[count/3]))
						elif info[0]=='beta':
							next_line = next_line.replace(info[1],str(beta[count/3]))
						content[k+1+count] = next_line+'\n'
						count += 1
						next_line = content[k+1+count].strip()
					except:
						break
		if '=' in line:
			if line.split('=')[0] == 'OUTPUT':
				new_dir = str(int(np.random.rand()*1000000))
				content[k] = 'OUTPUT=output'+new_dir+'/\n'
	if len(content) > 758:
		content = content[:758]
		content[-3:] = ['\n', '[Execute]\n', 'TASK=Miss_task\n']
	with open(mdir+'control'+new_dir+'.txt','w') as fo:
		for line in content:
			fo.write(line)
	#read bash file and copy to a new file
	bash_content = open(mdir+'hoang_run_simulation.sh','r').readlines()
	for i,line in enumerate(bash_content):
		#change job name
		if 'ef5_xxx' in line:
			bash_content[i] = line.replace('ef5_xxx','ef5_'+new_dir)
		if 'control.txt' in line:
			bash_content[i] = line.replace('control.txt','control'+new_dir+'.txt')
	with open(mdir+'hoang_run_'+new_dir+'.sh','w') as fo:
		for line in bash_content:
			fo.write(line)
	#Run model
	try:
		os.mkdir(mdir+'output'+new_dir)
		jj0=Popen(['qsub',mdir+'hoang_run_'+new_dir+'.sh'], stdout=PIPE, stderr=STDOUT)
		stdout0, nothing = jj0.communicate()
		job_id = stdout0.split(' ')[2]
		list_jobs = [job_id]
		while job_id in list_jobs:
			time.sleep(10)
			jj1=Popen(['qstat','-u','htranvie'], stdout=PIPE, stderr=STDOUT)
			stdout1, nothing = jj1.communicate()
			stdout1 = stdout1.split('\n')
			list_jobs = []
			for i in range(2,len(stdout1)):
				list_jobs.append(stdout1[i].split(' ')[0])
	except:
		print 'error'
		return
	#remove control and bash files
	try:
		os.remove(mdir+'hoang_run_'+new_dir+'.sh')
		os.remove(mdir+'control'+new_dir+'.txt')
	except:
		print 'error removing temporary control and bash files'
	#Evaluate with cloud-free flood maps
	list_CF = sorted(glob(wdir+'Water_depth/*.tif'))
	list_WD = sorted(glob(mdir+'output'+new_dir+'/*.tif'))
	CF_days = list(set([os.path.basename(x).split('_')[2][:8] for x in list_CF]))
	WD_days = list(set([os.path.basename(x).split('.')[1][:8] for x in list_WD]))
	day_to_compare = sorted([x for x in WD_days if x in CF_days])
	tot_res = pd.DataFrame()
	for d in day_to_compare:
		CF_files = [x for x in list_CF if d in x]
		WD_files = [x for x in list_WD if d in x]
		for idx,CF_file in enumerate(CF_files):
			#Read cloud-free flood map file
			ds_CF = gdal.Open(CF_file)
			geom_CF = ds_CF.GetGeoTransform()
			arr_CF = ds_CF.ReadAsArray()
			bb_CF = [geom_CF[0],geom_CF[3]-arr_CF.shape[0]*geom_CF[1],
						geom_CF[0]+arr_CF.shape[1]*geom_CF[1],geom_CF[3]]
			#Read output file from model
			WD_file = WD_files[idx]
			WD_resampled_file = wdir+'validation/clipped_files/'+os.path.basename(WD_file)
			#resample images to cloud-free flood map
			ResampleImage(WD_file,WD_resampled_file,geom_CF[1],bb_CF)
			#validation
			arr_WD = gdal.Open(WD_resampled_file).ReadAsArray()
			pd_res = Validation(d,arr_CF,arr_WD)
			tot_res = tot_res.append(pd_res, ignore_index=True)
	#return average FAR over the period
	return tot_res.loc[tot_res['Types']=='FAR'].iloc[:,2].mean(axis=0)


################################################################################


################################################################################
##  Sampling called from SCE
################################################################################

def SampleInputMatrix(nrows,npars,bu,bl,iseed,distname='randomUniform'):
    '''
    Create inputparameter matrix for nrows simualtions,
    for npars with bounds ub and lb (np.array from same size)
    distname gives the initial sampling ditribution (currently one for all parameters)

    returns np.array
    '''
    np.random.seed(iseed)
    x=np.zeros((nrows,npars))
    bound = bu-bl
    for i in range(nrows):
##        x[i,:]= bl + DistSelector([0.0,1.0,npars],distname='randomUniform')*bound  #only used in full Vhoeys-framework
        x[i,:]= bl + np.random.rand(1,npars)*bound
    return x


################################################################################
##    TESTFUNCTIONS TO CHECK THE IMPLEMENTATION OF THE SCE ALGORITHM
################################################################################
def testfunctn1(nopt,x):
    '''
    This is the Goldstein-Price Function
    Bound X1=[-2,2], X2=[-2,2]
    Global Optimum: 3.0,(0.0,-1.0)
    '''

    x1 = x[0]
    x2 = x[1]
    u1 = (x1 + x2 + 1.0)**2
    u2 = 19. - 14.*x1 + 3.*x1**2 - 14.*x2 + 6.*x1*x2 +3.*x2**2
    u3 = (2.*x1 - 3.*x2)**2
    u4 = 18. - 32.*x1 + 12.*x1**2 + 48.*x2 -36.*x1*x2 + 27.*x2**2
    u5 = u1 * u2
    u6 = u3 * u4
    f = (1. + u5) * (30. + u6)
    return f

def testfunctn2(nopt,x):
    '''
    %  This is the Rosenbrock Function
    %  Bound: X1=[-5,5], X2=[-2,8]; Global Optimum: 0,(1,1)
        bl=[-5 -5]; bu=[5 5]; x0=[1 1];
    '''

    x1 = x[0]
    x2 = x[1]
    a = 100.0
    f = a * (x2 - x1**2)**2 + (1 - x1)**2
    return f

def testfunctn3(nopt,x):
    '''3
    %  This is the Six-hump Camelback Function.
    %  Bound: X1=[-5,5], X2=[-5,5]
    %  True Optima: -1.031628453489877, (-0.08983,0.7126), (0.08983,-0.7126)
    '''
    x1 = x[0]
    x2 = x[1]
    f = (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2
    return f

def testfunctn4(nopt,x):
    '''4
    %  This is the Rastrigin Function
    %  Bound: X1=[-1,1], X2=[-1,1]
    %  Global Optimum: -2, (0,0)
    '''
    x1 = x[0]
    x2 = x[1]
    f = x1**2 + x2**2 - np.cos(18.0*x1) - np.cos(18.0*x2)
    return f

def testfunctn5(nopt,x):
    '''
    This is the Griewank Function (2-D or 10-D)
    Bound: X(i)=[-600,600], for i=1,2,...,10
    Global Optimum: 0, at origin
    '''
    if nopt==2:
        d = 200.0
    else:
        d = 4000.0

    u1 = 0.0
    u2 = 1.0
    for j in range(nopt):
        u1 = u1 + x[j]**2/d
        u2 = u2 * np.cos(x[j]/np.sqrt(float(j+1)))

    f = u1 - u2 + 1
    return f


################################################################################
##   FUNCTION CALL FROM SCE-ALGORITHM !!
################################################################################

def EvalObjF(npar,x,testcase=True,testnr=1):
    '''
    The SCE algorithm calls this function which calls the model itself
    (minimalisation of function output or evaluation criterium coming from model)
    and returns the evaluation function to the SCE-algorithm

    If testcase =True, one of the example tests are run
    '''
##    print 'testnummer is %d' %testnr

    if testcase==True:
        if testnr==1:
            return testfunctn1(npar,x)
        if testnr==2:
            return testfunctn2(npar,x)
        if testnr==3:
            return testfunctn3(npar,x)
        if testnr==4:
            return testfunctn4(npar,x)
        if testnr==5:
            return testfunctn5(npar,x)
    else:
		return Modrun(npar,x)          #Welk model/welke objfunctie/welke periode/.... users keuze!



