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
from glob import glob

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
	for i,line in enumerate(content):
		line = line.strip()
		if (line and line[0]=='[') and (line[-1]==']'):
			headers = line[1:-1].split(' ')
			if headers[0] == 'simpleinundationparamset':
				res1 = ParseLoopInfo(content,i)
	params = [[float(x), float(res1['beta'][j])] for j,x in enumerate(res1['alpha'])]
	return res1

#Objective HK
like_HK = lognorm(0.08)

#Define the ODE system given y, t, and a parameter set
def odefunc(params):
	CTfile = 'control'
	#Open control file
	content = open('control1.txt','r').readlines()
	res1 = ReadCT(content)
	for i in range(params.shape[0]/2):
		res1['alpha'][i] = str(params[2*i])
		res1['beta'][i] = str(params[2*i+1])
    #Modify control file based on new parameters
	for k,line in enumerate(content):
		line = line.strip()
		if (line and line[0]=='[') and (line[-1]==']'):
			headers = line[1:-1].split(' ')
			if headers[0] == 'simpleinundationparamset':
				for count in range(len(res1['beta'])*3):
					next_line = content[k+1+count]
					info = next_line.split('=')
					next_line = next_line.replace(info[1],res1[info[0]][count/3])
					content[k+1+count] = next_line+'\n'
		if '=' in line:
			if line.split('=')[0] == 'OUTPUT':
				new_dir = str(int(np.random.rand()*1000000))
				content[k] = 'OUTPUT=output'+new_dir+'/\n'
	
	if len(content) > 758:
		content = content[:758]
		content[-3:] = ['\n', '[Execute]\n', 'TASK=Miss_task\n']
	with open(CTfile+new_dir+'.txt','w') as fo:
		for line in content:
			fo.write(line)
	#Run model
	try:
		os.mkdir('output'+new_dir)
		jj=Popen(['ef5',CTfile+new_dir+'.txt'], stdout=PIPE, stderr=STDOUT)
		stdout, nothing = jj.communicate()
		return 'output'+new_dir
	except:
		return


#Define likelihood function to generate simulated data that corresponds to experimental time points.  
#This function should take as input a parameter vector
# (parameter values are in the order dictated by first argument to run_dream function below).
#The function returns a log probability value for the parameter vector given the experimental data.

def likelihood(parameter_vector):
	#Run model
	new_dir = odefunc(parameter_vector)
	if not new_dir:
		return -np.inf
	#Calculate HK score
	#flood inundation simulation
	list_simu = sorted(glob(new_dir+'/*.tif'))
	arr_simu = np.array([]).reshape(0,1494,1836)
	for fsimu in list_simu:
		arr1 = io.imread(list_simu[0])[44:-463,128:-637]
		arr_simu = np.vstack([arr1[np.newaxis,:,:],arr_simu])
	
	#mask basin
	basin_mask = io.imread('/pub/htranvie/second_year/hoang_thesis/Flood/elevation/mississippi_rasterized.tif')
	
	#flood observed
	list_obs = sorted(glob('../../Cloud_free/MOD_CF_200401*.npy'))[:8]
	arr_obs = np.array([]).reshape(0,1494,1836)
	for fobs in list_obs:
		arr2 = np.load(fobs)
		arr_obs = np.vstack([arr2[np.newaxis,:,:],arr_obs])
	
	arr_simu = arr_simu[:,basin_mask==1]
	arr_obs = arr_obs[:,basin_mask==1]
	hit_arr = np.sum(np.logical_and(arr_simu > 0.1, arr_obs==1),axis=1)
	miss_arr = np.sum(np.logical_and(np.logical_and(arr_simu>=0,arr_simu<=0.1), arr_obs ==1),
						axis=1)
	false_arr = np.sum(np.logical_and(arr_simu > 0.1, arr_obs==0),axis=1)
	corr_neg_arr = np.sum(np.logical_and(np.logical_and(arr_simu>=0,arr_simu<=0.1), arr_obs ==0						),axis=1)
	hk_arr = (hit_arr*corr_neg_arr-miss_arr*false_arr)/((hit_arr+miss_arr)*(false_arr+corr_neg_arr)).astype(np.float)
	with open('temp_params/'+new_dir+'.txt','w') as fo:
		np.savetxt(fo,np.hstack([hk_arr,parameter_vector]))
	print hk_arr
	#Calculate log probability contribution given simulated experimental values.
	
	logp_ctotal = np.sum(like_HK.logpdf(hk_arr))
	shutil.rmtree(new_dir)
	os.remove(new_dir.replace('output','control')+'.txt')
	#If simulation failed due to integrator errors, return a log probability of -inf.
	if np.isnan(logp_ctotal):
		logp_ctotal = -np.inf
		  
	return logp_ctotal


# Add vector of rate parameters to be sampled as unobserved random variables in DREAM with uniform priors.
  
original_params = np.ones((26*2))*0.2

#Set upper and lower limits for uniform prior to be 3 orders of magnitude above and below original parameter values.
lower_limits   = np.ones((26*2))*0.01

parameters_to_sample = SampledParam(uniform, loc=lower_limits, 
								scale=[1 if x%2 else 3 for x in range(26*2)])

#The run_dream function expects a list rather than a single variable
sampled_parameter_names = [parameters_to_sample]

niterations = 500
converged = False
total_iterations = niterations
nchains = 5

if __name__ == '__main__':

    #Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=niterations, nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1, model_name='EF5', verbose=True)
    
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
                                               nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
                                               history_thin=1, model_name='robertson_nopysb_dreamzs_5chain', verbose=True, restart=True)

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
    run_kwargs = {'parameters':sampled_parameter_names, 'likelihood':likelihood, 'niterations':100, 'nchains':nchains, 'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'EF5', 'verbose':True}
