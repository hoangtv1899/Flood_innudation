#!/data/apps/anaconda/2.7-4.3.1/bin/python

from pydream.core import run_dream
import numpy as np
from pydream.parameters import SampledParam
from scipy.stats import lognorm, uniform
from skimage import io
import os
from subprocess import Popen, PIPE, STDOUT
import inspect
import shutil
from pydream.convergence import Gelman_Rubin
from datetime import datetime
import pandas as pd
import time
import gdal
from glob import glob

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
	val_mod, type_mod = catValidation(arr_cf, arr_wd)
	results['Date'] = [d0]*7
	results['Types'] = type_mod
	results['Values'] = val_mod
	return results

def ResampleImage(img,img1,resolution,bb):
	Popen(['gdalwarp','-q','-overwrite','-dstnodata','-99','-co','COMPRESS=LZW','-tr',
			str(resolution),str(resolution),'-te',str(bb[0]),str(bb[1]),str(bb[2]),str(bb[3]),
			img,img1],
			stdout=PIPE, stderr=STDOUT).communicate()

#Define the ODE system given y, t, and a parameter set
def odefunc(params):
	wdir = '/pub/htranvie/second_year/hoang_thesis/Flood/'
	mdir = wdir+'ef5/mississippi_flood_map/'
	#Open control file
	content = open(mdir+'control1.txt','r').readlines()
	alpha, beta = [],[]
	for i in range(params.shape[0]/2):
		alpha.append(str(params[2*i]))
		beta.append(str(params[2*i+1]))
	new_dir = str(int(np.random.rand()*10e20))
	try:
		os.mkdir(mdir+'output'+new_dir)
	except:
		return
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
							next_line = next_line.replace(info[1],alpha[count/3])
						elif info[0]=='beta':
							next_line = next_line.replace(info[1],beta[count/3])
						content[k+1+count] = next_line+'\n'
						count += 1
						next_line = content[k+1+count].strip()
					except:
						break
		if '=' in line:
			if line.split('=')[0] == 'OUTPUT':
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
		jj0=Popen(['qsub',mdir+'hoang_run_'+new_dir+'.sh'], stdout=PIPE, stderr=STDOUT)
		stdout0, nothing = jj0.communicate()
		# print stdout0
		job_id = stdout0.split('Your job ')[1].split(' ')[0]
		list_jobs = [job_id]
		mstart = time.time()
		while job_id in list_jobs:
			#print 'model #'+new_dir+' is running...'
			time.sleep(15)
			jj1=Popen(['qstat','-u','htranvie'], stdout=PIPE, stderr=STDOUT)
			stdout1, nothing = jj1.communicate()
			# print stdout1
			stdout1 = stdout1.split('\n')
			list_jobs = []
			for i in range(2,len(stdout1)):
				list_jobs.append(stdout1[i].split(' ')[0])
		mend = time.time()
		# print 'total time for running model #'+new_dir+' is: '+str(mend-mstart)+' seconds'
		os.remove(mdir+'hoang_run_'+new_dir+'.sh')
		os.remove(mdir+'control'+new_dir+'.txt')
		os.remove(mdir+'ef5_'+new_dir+'.e'+job_id)
		os.remove(mdir+'ef5_'+new_dir+'.o'+job_id)
		return new_dir
	except Exception as ex:
		print 'error running model # '+new_dir
		print ex.__doc__
		print ex.message
		os.remove(mdir+'hoang_run_'+new_dir+'.sh')
		os.remove(mdir+'control'+new_dir+'.txt')
		try:
			os.remove(mdir+'ef5_'+new_dir+'.e'+job_id)
			os.remove(mdir+'ef5_'+new_dir+'.o'+job_id)
		except:
			return
		return

#Define likelihood function to generate simulated data that corresponds to experimental time points.  
#This function should take as input a parameter vector
# (parameter values are in the order dictated by first argument to run_dream function below).
#The function returns a log probability value for the parameter vector given the experimental data.

def likelihood(parameter_vector):
	#Objective FAR
	like_FAR = lognorm(0.98)
	#Run model
	new_dir = odefunc(parameter_vector)
	if not new_dir:
		return -np.inf
	#working directory
	wdir = '/pub/htranvie/second_year/hoang_thesis/Flood/'
	mdir = wdir+'ef5/mississippi_flood_map/'
	#Evaluate with cloud-free flood maps
	vstart = time.time()
	list_Landsat = sorted(glob(wdir+'Landsat/*.tif'))
	list_CF = sorted(glob(wdir+'Water_depth/*.tif'))
	list_WD = sorted(glob(mdir+'output'+new_dir+'/*.tif'))
	CF_days = list(set([os.path.basename(xx).split('_')[2][:8] for xx in list_CF]))
	WD_days = list(set([os.path.basename(xx).split('.')[1][:8] for xx in list_WD]))
	day_to_compare = sorted([xx for xx in WD_days if xx in CF_days])
	os.mkdir(wdir+'validation/clipped_files/'+new_dir)
	tot_res = pd.DataFrame()
	for d in day_to_compare:
		Landsat_file = [xx for xx in list_Landsat if d in xx][0]
		CF_files = [xx for xx in list_CF if d in xx]
		WD_files = [xx for xx in list_WD if d in xx]
		if len(CF_files) == len(WD_files):
			for idx,CF_file in enumerate(CF_files):
				#Read cloud-free flood map file
				ds_CF = gdal.Open(CF_file)
				geom_CF = ds_CF.GetGeoTransform()
				arr_CF = ds_CF.ReadAsArray()
				bb_CF = [geom_CF[0],geom_CF[3]-arr_CF.shape[0]*geom_CF[1],
							geom_CF[0]+arr_CF.shape[1]*geom_CF[1],geom_CF[3]]
				#Read output file from model
				WD_file = WD_files[idx]
				#Create a temporary clipped directory
				WD_resampled_file = wdir+'validation/clipped_files/'+new_dir+'/'+os.path.basename(WD_file)
				L_resampled_file = wdir+'validation/clipped_files/'+new_dir+'/'+os.path.basename(Landsat_file)
				try:
					#resample Landsat image to cloud-free flood map
					ResampleImage(Landsat_file,L_resampled_file,geom_CF[1],bb_CF)
					arr_Landsat = gdal.Open(L_resampled_file).ReadAsArray()
					#resample images to cloud-free flood map
					ResampleImage(WD_file,WD_resampled_file,geom_CF[1],bb_CF)
					arr_WD = gdal.Open(WD_resampled_file).ReadAsArray()
					final_WD = ((arr_WD+arr_CF)>3.5).astype(np.int)
					#validation
					pd_res = Validation(d,arr_Landsat,final_WD)
					tot_res = tot_res.append(pd_res, ignore_index=True)
				except:
					continue
		else:
			CF_file = CF_files[0]
			WD_file = WD_files[0]
			ds_CF = gdal.Open(CF_file)
			geom_CF = ds_CF.GetGeoTransform()
			arr_CF = ds_CF.ReadAsArray()
			bb_CF = [geom_CF[0],geom_CF[3]-arr_CF.shape[0]*geom_CF[1],
						geom_CF[0]+arr_CF.shape[1]*geom_CF[1],geom_CF[3]]
			#Create a temporary clipped directory
			WD_resampled_file = wdir+'validation/clipped_files/'+new_dir+'/'+os.path.basename(WD_file)
			L_resampled_file = wdir+'validation/clipped_files/'+new_dir+'/'+os.path.basename(Landsat_file)
			try:
				#resample Landsat image to cloud-free flood map
				ResampleImage(Landsat_file,L_resampled_file,geom_CF[1],bb_CF)
				arr_Landsat = gdal.Open(L_resampled_file).ReadAsArray()
				#resample images to cloud-free flood map
				ResampleImage(WD_file,WD_resampled_file,geom_CF[1],bb_CF)
				arr_WD = gdal.Open(WD_resampled_file).ReadAsArray()
				final_WD = ((arr_WD+arr_CF)>3.5).astype(np.int)
				#validation
				pd_res = Validation(d,arr_Landsat,final_WD)
				tot_res = tot_res.append(pd_res, ignore_index=True)
			except:
				continue
	vend = time.time()
	# print 'total time for validation of '+new_dir+' is '+str(vend-vstart)+' seconds'
	#remove the temporary waterdepth directory
	os.system('rm -rf '+mdir+'output'+new_dir)
	os.system('rm -rf '+wdir+'validation/clipped_files/'+new_dir)
	#return average FAR over the period
	res_arr = tot_res.loc[tot_res['Types']=='FAR'].iloc[:,2].tolist()
	with open(mdir+'temp_params/'+new_dir+'.txt','w') as fo:
		np.savetxt(fo,np.hstack([np.asarray(res_arr),parameter_vector]))
	#Calculate log probability contribution given simulated experimental values.
	logp_ctotal = np.sum(like_FAR.logpdf(res_arr))
	#If simulation failed due to integrator errors, return a log probability of -inf.
	if np.isnan(logp_ctotal):
		logp_ctotal = -np.inf
	return -logp_ctotal

# Add vector of rate parameters to be sampled as unobserved random variables in DREAM with uniform priors.
original_params = np.ones((18*2))*0.2

#Set upper and lower limits for uniform prior to be 3 orders of magnitude above and below original parameter values.
lower_limits   = np.ones((18*2))*0.001

parameters_to_sample = SampledParam(uniform, loc=lower_limits, 
								scale=[2 if x%2 else 5 for x in range(18*2)])

#The run_dream function expects a list rather than a single variable
sampled_parameter_names = [parameters_to_sample]

niterations = 100
converged = False
total_iterations = niterations
nchains = 5

if __name__ == '__main__':

    #Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=niterations, nchains=nchains, multitry=8, gamma_levels=4, adapt_gamma=True, history_thin=1, model_name='EF5', verbose=True, parallel=True)
    
    #Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save('EF5_sampled_params_chain_'+str(chain)+'_'+str(total_iterations), sampled_params[chain])
        np.save('EF5_logps_chain_'+str(chain)+'_'+str(total_iterations), log_ps[chain])

    # Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ', total_iterations, ' GR = ', GR)
    np.savetxt('EF5_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)

    old_samples = sampled_params
    if np.any(GR > 1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations

            sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=niterations,
                                               nchains=nchains, multitry=8, gamma_levels=4, adapt_gamma=True,
                                               history_thin=1, model_name='EF5', verbose=True, restart=True, parallel=True)

            for chain in range(len(sampled_params)):
                np.save('EF5_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
                            sampled_params[chain])
                np.save('EF5_logps_chain_' + str(chain) + '_' + str(total_iterations),
                            log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ', total_iterations, ' GR = ', GR)
            np.savetxt('EF5_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)

            if np.all(GR < 1.2):
                converged = True

else:
    run_kwargs = {'parameters':sampled_parameter_names, 'likelihood':likelihood, 'niterations':100, 'nchains':nchains, 'multitry':8, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'EF5', 'verbose':True, 'parallel':True, 'save_history':True,
	'history_file':True}
