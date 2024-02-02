# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 14:29:44 2021

@author: Ting Li
"""

import numpy as np
import scipy as sp
import datetime
import multiprocessing as mp
import pandas as pd
import scipy.linalg
from random import choices
import copy

import time
from time import *
import sys
if path not in sys.path:
    sys.path.append(path)    
from settings_multiple_S_simu import QRVCM

# ##################### Data settings #######################

import warnings
warnings.filterwarnings("ignore")

###################### Estimate the coefficients based on real data #################

        ## y_it: n*m  array
        ## A_it: n*m  array
        ## S_it: dictionary {'S1': S_it1, ..., 'Sd': S_itd}, each S_itj is an n*m array
        ## time_points: t_j 
        ## hc: bandwidth for the coefficients
        ## tau: quantile level
########################################################################
def generate_temporal_RealData_based( y_it, S_it, coes_0, tau, delta_DE, delta_IE, NN, TI  ):
    number_of_dates, nsamples_per_day = y_it.shape 
    d = len(S_it)
    covariate_index = [key for key, value in S_it.items()] 
    
    coe_y_smoothed_0 = coes_0['coe_y_smoothed_0']
    resid_vY_0 = coes_0['resid_vY_0']
    coe_s_smoothed_0 = coes_0['coe_s_smoothed_0']
    resid_mS_0 = coes_0['resid_mS_0']
    reg_mat_0 = coes_0['reg_mat_0']
    
    resid_vY0_matrix = resid_vY_0.reshape(number_of_dates,  nsamples_per_day, order='F') 

    resid_vY0_matrix = resid_vY0_matrix - np.quantile(resid_vY0_matrix, q=tau, axis=0)

    resid_s0_matrix={}

    for xid in range(d):
        resid_vS0_matrix = resid_mS_0[:, xid].reshape(number_of_dates,  nsamples_per_day-1, order='F' )
        resid_vS0_matrix =  resid_vS0_matrix - np.quantile( resid_vS0_matrix, q=tau, axis=0)
        resid_s0_matrix[ covariate_index[xid]  ] = resid_vS0_matrix
    
    select_dates = choices( list(range(number_of_dates)),  k=NN )  ## for fitted
    
    select_samples = list(np.hstack( [number_of_dates*sample + np.array(select_dates)  for sample in range(nsamples_per_day)] ) )
    
    reg_mat_select = reg_mat_0[select_samples, :]
    #########
    
    
    positions =  choices( list(range(number_of_dates)),  k=NN ) ## for residulas
    positions_s = [choices( list(range(number_of_dates)),  k=NN ) for _ in range(d)] ## for residulas
    
    ################ select residuals ###########    
    resid_vY_matrix_select = resid_vY0_matrix[positions, :]
    resid_vY_select = resid_vY_matrix_select.reshape(-1, 1, order='F')
    
    #### generate treatment #####  
    AB_vector = np.tile( np.repeat( [1, 0], TI ), nsamples_per_day// TI //2 )
    BA_vector = np.tile( np.repeat( [0, 1], TI ), nsamples_per_day// TI// 2 ) 
    vec =np.hstack([AB_vector, BA_vector])
    
    sample_A_it =np.array( np.tile(vec, NN//2) ).reshape(NN, nsamples_per_day)
    
    #### generate sample_Y_it and sample_S_it ###############
    gamma_y = np.quantile(y_it, q=tau, axis=0)*delta_DE

    
    fitted_vY_select  = ( reg_mat_0[select_dates, :].dot( coe_y_smoothed_0.iloc[:, 0].values )).reshape(-1,1)  \
        + ( sample_A_it[:, 0]*gamma_y[0] ).reshape(-1,1) +  resid_vY_select[np.arange(0, NN)]
        
    resid_mS_select =[]
    Gamma_s =[]
    for xid in range(d):
        resid_vS_tmp = resid_s0_matrix[ covariate_index[xid]  ][positions_s[xid],:]
        resid_mS_select.append( resid_vS_tmp.reshape(-1,1, order='F') )
        Gamma_s.append( np.quantile(S_it[ covariate_index[xid] ], q=tau, axis=0)*delta_IE )

    resid_mS_select = np.hstack(resid_mS_select)
    
    
    for t in np.delete( range(m),0):
        reg_s_diag_mat = reg_mat_select[np.arange(NN*(t-1), t*NN),:]              
        for xid in range(d): 
            index = covariate_index[xid]                
            reg_mat_select[np.arange(NN*t, (t+1)*NN), xid+1 ] =  ( reg_s_diag_mat.dot( coe_s_smoothed_0[index].loc[:, 't='+str(t-1)].values )) \
                + sample_A_it[:, t-1]*Gamma_s[xid][t-1] + resid_mS_select[np.arange(NN*(t-1), t*NN), xid] 
            
        fitted_vY_bt = ( reg_mat_select[np.arange(NN*t, (t+1)*NN), :].dot( coe_y_smoothed_0.iloc[:, t].values ) ).reshape(-1,1)  \
                + (sample_A_it[:, t]*gamma_y[t] ).reshape(-1,1)+ resid_vY_select[np.arange(NN*t, (t+1)*NN)]
           
        fitted_vY_select = np.vstack( (fitted_vY_select, fitted_vY_bt))

    y_it_select = fitted_vY_select.reshape(NN, nsamples_per_day, order='F')
                
    S_it_select={}
    for xid in range(d): 
        S_it_select[covariate_index[xid] ] = ( reg_mat_select[:, xid+1]).reshape(NN, nsamples_per_day, order='F')
    
    return y_it_select, S_it_select,  sample_A_it




######################## settings ##################################################

################# settings ##########

hc=hcb=0   ### only when the time pints are equal contribute to the estimate


delta_IE=0  
delta_DE=0
NN=40
TI=1
tau=0.5

#################
boot_times =500    # bootstrap time
Run=500     # simulation runs

num_cores =10   # number of cores in parellel computing


#########################################################
df = pd.read_csv('simu_demo_data.csv', index_col=['date','time']) # generate simulation data based on this dataset


dates = len(df.index.get_level_values(0).unique())  # number of date
m = len(df.index.get_level_values(1).unique())  # number of time

y_it = df.loc[:,'y'].values.reshape(dates, m)
S_it ={}
S_it['S1'] = df.loc[:,'S1'].values.reshape(dates, m)
S_it['S2'] = df.loc[:,'S2'].values.reshape(dates, m)

time_points = np.array( range(m) )/m

A_it_tmp = np.zeros((dates, m))


model0 = QRVCM( y_it, A_it_tmp, S_it, time_points, tau, hc, hcb )
coes_0 = model0.estimate_withoutA()


####################### true value of gamma_y and Gamma_s ##########
gamma_y = np.quantile(y_it, q=tau, axis=0)*delta_DE

Gamma_s1 = np.quantile(S_it[ 'S1'], q=tau, axis=0)*delta_IE
Gamma_s2 = np.quantile(S_it[ 'S2'], q=tau, axis=0)*delta_IE

Gamma_s={}
for t in range(m):
    Gamma_s['t=' + str(t)] = np.array( [Gamma_s1[t], Gamma_s2[t] ]).reshape(-1,1)

############ Estimated from real data ########################
beta_1 = coes_0['coe_y_smoothed_0'].drop(['Intercept'], axis=0).values 

covariate_index = [key for key,value in S_it.items()]
d=len(covariate_index)
phi={}
for t in np.delete(range(m), m-1):
    phi['t='+str(t)] = None
    for index_response in covariate_index:
        if index_response==covariate_index[0]:
            phi['t='+str(t)] =coes_0['coe_s_smoothed_0'][index_response].drop(['Intercept']).loc[:, 't='+str(t)].values.reshape(1,d) 
        else:
            phi['t='+str(t)] = np.vstack((phi['t='+str(t)], coes_0['coe_s_smoothed_0'][index_response].drop(['Intercept']).loc[:, 't='+str(t)].values.reshape(1,d)   ))
 
QDE_true, QIE_true= model0.QDE_QTE_calculate( time_start=0, time_end=m, gamma_y=gamma_y, Gamma_s=Gamma_s, phi=phi, beta_1=beta_1)

########### true coes ######################

coe_y_true = np.vstack( (coes_0['coe_y_smoothed_0'].loc['Intercept',:].values.reshape(1, m), gamma_y.reshape(1,m), beta_1 ) )

coe_s1_true = copy.deepcopy( coes_0['coe_s_smoothed_0']['S1'] )
coe_s1_true.loc['A',:] =np.delete( Gamma_s1,m-1)
coe_s1_true['order']=[1,3,4,2]
coe_s1_true=coe_s1_true.sort_values(by='order')
coe_s1_true=coe_s1_true.drop(columns=['order'])

coe_s1_true['t=m-1']=''


coe_s2_true = copy.deepcopy( coes_0['coe_s_smoothed_0']['S2'] )
coe_s2_true.loc['A',:] =np.delete( Gamma_s2,m-1)
coe_s2_true['order']=[1,3,4,2]
coe_s2_true=coe_s2_true.sort_values(by='order')
coe_s2_true=coe_s2_true.drop(columns=['order'])

coe_s2_true['t=m-1']=''

#############################
def Simulation_run(R):

    Simu_results =[]

    for r in R:
       sp.random.seed(r) 
       print('r='+str(r))
       y_it_select, S_it_select, sample_A_it = generate_temporal_RealData_based( y_it, S_it, coes_0, tau, delta_DE, delta_IE, NN, TI  )
       model = QRVCM( y_it_select, sample_A_it,  S_it_select, time_points, tau, hc, hcb )
       coe_y_smoothed, resid_vY, coe_s_smoothed, resid_mS=model.estimate()
       QDE, QIE= model.QDE_QTE_calculate(time_start=0, time_end=m)

       run_start=time()
       pvalues, mean_stat_QIE_b, sd_stat_QIE_b  = model.testing(B=boot_times, time_start=0, time_end=m)      
       run_end = time()

       print('Boostrap time:', run_end-run_start)

       run_time = (run_end-run_start)/60  # in minutes


###### calculate MSEs of coefficients ####
       coe_y_mse = np.average(( np.array( model.coe_y_smoothed) - coe_y_true)**2, axis=1)
       coe_s1_mse = np.average(( np.array( model.coe_s_smoothed['S1']) - coe_s1_true)**2, axis=1)
       coe_s2_mse = np.average(( np.array( model.coe_s_smoothed['S2']) - coe_s2_true)**2, axis=1)
   
       results_tmp =np.array( [ [QDE, QIE, coe_y_mse[0], coe_y_mse[1], coe_y_mse[2], coe_y_mse[3],
                           coe_s1_mse[0], coe_s1_mse[1], coe_s1_mse[2],coe_s1_mse[3],
                           coe_s2_mse[0], coe_s2_mse[1], coe_s2_mse[2],coe_s2_mse[3],
                           mean_stat_QIE_b, sd_stat_QIE_b , run_time] ] )
       
       results_tmp  = np.hstack( (np.array(pvalues).reshape(1, 8), results_tmp) )

   
       Simu_results.append( results_tmp)

    return np.vstack( Simu_results )


if __name__ == '__main__':
    
    
    start_t = datetime.datetime.now()

    print("Use: " + str(num_cores) + " Cores")
    pool = mp.Pool(num_cores)
    
    
    Run_each_core = Run // num_cores
    
    
    tasks= [list(np.arange((num_a)*Run_each_core, (num_a +1)*Run_each_core)) for num_a in list(range(num_cores) )]  
    
    results = pool.map(Simulation_run,tasks) 
    
    pool.close()
    
    ############### merge results ###############
    results = np.vstack( results )

    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("Elasped: " + "{:.2f}".format(elapsed_sec) + " s")
    

######### calculate statistics ####
   
    results_all = np.hstack( (results, (( results[:, 8] - QDE_true )**2).reshape(-1,1),  (( results[:, 9] - QIE_true )**2).reshape(-1,1)  ) )## QDE_mse ## QIE_mse
    
    indicators = results_all[:, [0,1,2,3,4,5,6,7] ] < 0.05
    
    results_all = np.hstack((results_all, indicators) )
    
    Mean_results = np.mean( results_all, axis=0 ).reshape(1, -1)
    sd_results = np.std( results_all, axis=0 ).reshape(1, -1)
    
    results_all = np.vstack( (results_all, Mean_results, sd_results) )
    
    col_names=['pvalue_QTE_twoSides','pvalue_QDE_twoSides', 'pvalue_QIE_twoSides', 'pvalue_normal_QDE_twoSides',
               'pvalue_QTE_RightSide','pvalue_QDE_RightSide', 'pvalue_QIE_RightSide', 'pvalue_normal_QDE_RightSide',
               'QDE', 'QIE', 
               'coe_beta0_mse', 'coe_gamma_y_mse', 'coe_beta11_mse', 'coe_beta12_mse',
                           'coe_phi01_mse', 'coe_Gamma_s1_mse', 'coe_phi11_mse', 'coe_phi12_mse',
                           'coe_phi02_mse', 'coe_Gamma_s2_mse', 'coe_phi21_mse', 'coe_phi22_mse',
                           'mean_stat_QIE_b', 'sd_stat_QIE_b','run_time', 'QDE_mse', 'QIE_mse','QTE_indicator_twoSides',
                           'QDE_indicator_twoSides', 'QIE_indicator_twoSides','QDE_norm_indicator_twoSides',
                           'QTE_indicator_RightSide','QDE_indicator_RightSide', 'QIE_indicator_RightSide','QDE_norm_indicator_RightSide']
    
    index_name = [ 'Run'+str(s) for s in range(Run)]
    
    index_name.append('Mean')
    index_name.append('Sd')
    
    results_pd = pd.DataFrame( results_all, columns=col_names, index=index_name )
 ########## results output ######################
    folder_path ='./ResidMultiple_NN_'+str(NN)+'_TI_'+str(TI)
    
    import os
    if not os.path.exists(folder_path):
       os.makedirs(folder_path)
       
    file_name = folder_path + '/delta_IE_' +str(delta_DE)+'_deltaIE_'+str(delta_IE)+'_tau_'+str(tau)+ '_h_'+str(hc)+'.csv'    
    
    results_pd.to_csv(file_name)