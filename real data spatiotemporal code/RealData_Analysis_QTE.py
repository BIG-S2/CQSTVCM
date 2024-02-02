import numpy as np
import scipy as sp
import datetime
import multiprocessing as mp
import pandas as pd
import scipy.linalg
from random import choices
import copy

import time
path='./RealData'
import sys
if path not in sys.path:
    sys.path.append(path)    
from Spatial_S_without_barA import SQRVCM_without_barA  # it can exclude bar_A_it
from Spatial_S_without_barA import ker 

# ##################### Data settings #######################

import warnings
warnings.filterwarnings("ignore")

R = 12
from time import *


#########################################################
df = pd.read_csv('data_preprocess_from_8am_scaled.csv', index_col=['region','date','time'])

regions = len(df.index.get_level_values(0).unique()) 
dates = len(df.index.get_level_values(1).unique())  # number of date
m = len(df.index.get_level_values(2).unique())  # number of time


y_rit = df.loc[:,'y'].values.reshape(regions,dates, m) / np.std(df.loc[:, 'y'])
S_rit ={}
S_rit['S1'] = df.loc[:,'S1'].values.reshape(regions, dates, m) / np.std(df.loc[:, 'S1'])


time_points = np.array( range(m) )/m

A_rit = df.loc[:,'A'].values.reshape(regions, dates, m)

######################################################################

hrz = np.array([392,397,416,385,399,368,423,397,405,405,412,401]) # Latitude for the regions
hrz = (hrz-min(hrz))/(max(hrz) - min(hrz))

vtz = np.array([591,892,799,687,461,626,558,763,521,773,694,654]) # longitude for the regions
vtz = (vtz-min(vtz))/(max(vtz) - min(vtz))

hr=2*np.std(hrz)
vr=2*np.std(vtz)
ker_mat_r = ker( np.array(hrz ).reshape(-1,1) - np.array(hrz ).reshape(1,-1), hr)*ker( np.array(vtz ).reshape(-1,1) - np.array(vtz ).reshape(1,-1), vr)

ker_mat_r = ker_mat_r / ker_mat_r.sum(axis=0).T


hc=hcb=5/m
######################################################################
taus = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for tau in taus:
    model = SQRVCM_without_barA( y_rit, A_rit, S_rit,  time_points,  tau,  ker_mat_r,hc, hcb )
    coe_y_smoothed_region, coe_s_smoothed_region, resid_vY, resid_mS=model.estimate_allRegions()
    QDE, QIE= model.QDE_QTE_AllRegion(time_start=0, time_end=m)
    
    run_start=time()
    pvalues, mean_stat_QIE_b, sd_stat_QIE_b  = model.testing(B=500, time_start=0, time_end=m)
    run_end = time()
    
    print('Boostrap time:', run_end-run_start)
    ########### Results save ########################
    pvalues['QDE_QIE_value'] =[QDE, QIE]
    
    df_output_y = { 'y':[], 'fitted_y':[], 'resid_y':[]}
    
    keys = [key for key in resid_vY.keys()]
    
    
    for r in range(regions):
        resid_temp = resid_vY[ keys[r] ] #11... n1; 12... n2,;... 1m.... nm.
        fitted_temp = model.fitted_vY[ keys[r] ]
        
        df_output_y['y'].append( y_rit[r].reshape(-1,1, order='F') )
        df_output_y['fitted_y'].append(fitted_temp )
        df_output_y['resid_y'].append(resid_temp )
        
    df_output_y['y'] = np.vstack(df_output_y['y']).reshape(-1,)    
    df_output_y['fitted_y'] = np.vstack(df_output_y['fitted_y'] ).reshape(-1,)  
    df_output_y['resid_y'] = np.vstack(df_output_y['resid_y'] ).reshape(-1,)  
    
    df_output_y=pd.DataFrame( df_output_y, index=[val for val in keys for i in range( resid_temp.shape[0] ) ] )
    
    ######################
    df_output_S1 = { 'S1':[], 'fitted_S1':[], 'resid_S1':[]}
    
    
    for r in range(regions):
        resid_S1_temp = resid_mS[ keys[r] ] #11... n1; 12... n2,;... 1m.... nm.
        fitted_S1_temp = model.fitted_mS[ keys[r] ]
        
        df_output_S1['S1'].append( S_rit['S1'][r, :,np.arange(1,m)].reshape(-1,1) )
        df_output_S1['fitted_S1'].append(fitted_S1_temp )
        df_output_S1['resid_S1'].append(resid_S1_temp )
        
    df_output_S1['S1'] = np.vstack(df_output_S1['S1']).reshape(-1,)    
    df_output_S1['fitted_S1'] = np.vstack(df_output_S1['fitted_S1'] ).reshape(-1,)  
    df_output_S1['resid_S1'] = np.vstack(df_output_S1['resid_S1'] ).reshape(-1,)  
    
    df_output_S1=pd.DataFrame( df_output_S1, index=[val for val in keys for i in range( resid_S1_temp.shape[0] ) ] )
    
    ######################
    file_name='./Results/'+'Est_testing_'+str(tau)+'.csv'
    file_name_1 ='./Results/'+'y_tau_'+str(tau)+'.csv'
    file_name_2 ='./Results/'+'S1_tau_'+str(tau)+'.csv'
    
    pvalues.to_csv(file_name) 
    df_output_y.to_csv( file_name_1 )   
    df_output_S1.to_csv( file_name_2 )






