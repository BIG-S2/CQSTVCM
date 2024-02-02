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
from linear_multiple_S import ker 
from linear_multiple_S import LRVCM

# ##################### Data settings #######################

import warnings
warnings.filterwarnings("ignore")



#### Input data #########################################
#########################################################
df = pd.read_csv('simu_demo_data.csv', index_col=['date','time'])


dates = len(df.index.get_level_values(0).unique())  # number of date
m = len(df.index.get_level_values(1).unique())  # number of time

y_it = df.loc[:,'y'].values.reshape(dates, m)
S_it ={}
S_it['S1'] = df.loc[:,'S1'].values.reshape(dates, m)
S_it['S2'] = df.loc[:,'S2'].values.reshape(dates, m)
A_it = df.loc[:,'A'].values.reshape(dates, m)


time_points = np.array( range(m) )/m



####################### tuning parameters settings ###############################################

hc=hcb=5/m   ### only when the time pints are equal contribute to the estimate


from time import *
#################
boot_times =500   # bootstrap time



######################################################################
model_linear = LRVCM(  y_it, A_it, S_it,time_points, hc, hcb )
coe_y_smoothed, resid_vY, coe_s_smoothed, resid_mS=model_linear.estimate()
QDE, QIE= model_linear.QDE_QTE_calculate(time_start=0, time_end=m)

run_start=time()
pvalues, mean_stat_QIE_b, sd_stat_QIE_b  = model_linear.testing(B=boot_times, time_start=0, time_end=m)
run_end = time()


print('Boostrap time:', run_end-run_start)
########### Results save ########################
pvalues['QDE_QIE_value'] =[QDE, QIE]

df_output_y = { 'y':y_it.reshape(-1,1, order='F'), 'fitted_y':model_linear.fitted_vY, 'resid_y':resid_vY}

df_output_y['y'] = df_output_y['y'].reshape(-1,)
df_output_y['fitted_y'] = df_output_y['fitted_y'].reshape(-1,)
df_output_y['resid_y']  =  df_output_y['resid_y'].reshape(-1,) 

df_output_y=pd.DataFrame( df_output_y, index=df.index )

######################
df_output_S1 = { 'S1':S_it['S1'][:,np.arange(1,m)].reshape(-1,1,order='F'),
                'fitted_S1':model_linear.fitted_mS[:, 0], 'resid_S1':resid_mS[:, 0]}
    
df_output_S1['S1'] = df_output_S1['S1'].reshape(-1,)    
df_output_S1['fitted_S1'] = df_output_S1['fitted_S1'].reshape(-1,)  
df_output_S1['resid_S1'] = df_output_S1['resid_S1'].reshape(-1,)  

df_output_S1['date'] = np.repeat( range(dates), m-1)
df_output_S1['time'] = np.tile(np.arange(1, m), dates)

df_output_S1=pd.DataFrame( df_output_S1 )
df_output_S1=df_output_S1.set_index(['date','time'])


######################
df_output_S2 = { 'S2':S_it['S2'][:,np.arange(1,m)].reshape(-1,1,order='F'),
                'fitted_S2':model_linear.fitted_mS[:, 1], 'resid_S2':resid_mS[:, 1]}
    
df_output_S2['S2'] = df_output_S2['S2'].reshape(-1,)    
df_output_S2['fitted_S2'] = df_output_S2['fitted_S2'].reshape(-1,)  
df_output_S2['resid_S2'] = df_output_S2['resid_S2'].reshape(-1,)  

df_output_S2['date'] = np.repeat( range(dates), m-1)
df_output_S2['time'] = np.tile(np.arange(1, m), dates)

df_output_S2=pd.DataFrame( df_output_S2 )
df_output_S2=df_output_S2.set_index(['date','time'])

############

df_output_S=pd.concat([df_output_S1, df_output_S2], axis=1)

######### estimates of coes ########

coe_s_smoothed['S1']['t='+str(m-1)] = ''
coe_s_smoothed['S2']['t='+str(m-1)] = ''

coe_y_smoothed.columns=coe_s_smoothed['S1'].columns

coes_all = pd.concat([coe_y_smoothed, coe_s_smoothed['S1'], coe_s_smoothed['S2']])

coes_all.index=['Y-Intercept','Y-A','Y-S1','Y-S2',
                'S1-Intercept','S1-A','S1-S1','S1-S2',
                'S2-Intercept','S2-A','S2-S1','S2-S2']

######################
file_name='./Results/'+'Testing_linear.csv'
file_name_1 ='./Results/'+'y_tau_linear.csv'
file_name_2 ='./Results/'+'S_tau_linear.csv'
file_name_3 ='./Results/'+'Est_tau_linear.csv'

pvalues.to_csv(file_name) 
df_output_y.to_csv( file_name_1 )   
df_output_S.to_csv( file_name_2 )
coes_all.to_csv(file_name_3)



