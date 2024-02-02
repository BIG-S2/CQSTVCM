import numpy as np
import scipy as sp
from scipy.stats import norm
import datetime
import multiprocessing as mp


import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import copy
#import scipy.linalg
from random import choices




import time

##############################################
def ker(t, h):
    return np.exp(-t**2/h**2)

class LRVCM(object):
    def __init__(self, y_it, A_it, S_it,time_points,  hc, hcb):
        
        ## y_it: n*m  array
        ## A_it: n*m  array
        ## S_it: dictionary {'S1': S_it1, ..., 'Sd': S_itd}, each S_itj is an n*m array
        ## time_points: t_j 
        ## hc: bandwidth for the coefficients
        ## tau: quantile level
        
        # main args
        self.y_it = y_it
        self.S_it = S_it
        self.A_it = A_it
        self.d = len(S_it)
        self.hcb=hcb
        
        self.vY = y_it.reshape(-1, 1, order='F') # vector with y_11, ..., y_n1, ..., y_1m, ...,y_nm
        self.vA = A_it.reshape(-1, 1, order='F')
        
        
        self.n = y_it.shape[0]
        self.m = y_it.shape[1]
        self.nm =self.n*self.m
        self.time_points = time_points 
        self.hc = hc

        
        self.covariate_index=list()
        self.mS = None   ## nm*d matrix
        for key, value in self.S_it.items():
            self.covariate_index.append(key)
            if self.mS is None:
                self.mS=value.reshape(-1, 1, order='F')
            else:
                self.mS = np.hstack( (self.mS, value.reshape(-1, 1, order='F') ) )
                
                
        ### generate kernel matrix grids ######
        if self.hc==0:
            self.ker_mat_y = np.identity(self.m)  ## the orginal estimator without smoothing
            self.ker_mat_s = np.identity(self.m -1)
        else:
            self.ker_mat_y = ker( np.array(self.time_points ).reshape(-1,1) - np.array(self.time_points ).reshape(1,-1), self.hc)
            time_points_s = np.delete(self.time_points, self.m-1)
            self.ker_mat_s = np.identity(self.m -1)

            
        ### estimates ###
        self.coe_y_smoothed = None
        self.coe_s_smoothed = None
        
        self.fitted_vY = None
        self.fitted_mS = None
        self.reg_mat_all = None
        
    def estimate(self, y_it=None, S_it=None, A_it=None, hc=None):
        if y_it is None: y_it = self. y_it
        if S_it is None: S_it = self.S_it
        if A_it is None: A_it = self.A_it
        if hc is None: hc=self.hc
            
            ###### pointwise estimation for DE ################
        coe_y_all=pd.DataFrame()
            
        for  t in range(self.m):
                
            #### dataframe for DE quantile regression ###
            data_t = {'y':y_it[:, t], 'A':A_it[:, t] }
                
            reg_term='y~A'
            for index in self.covariate_index:
                data_t[index]=S_it[index][:, t]
                reg_term=reg_term + '+'+index
                
            data_t=pd.DataFrame(data_t)    
            temp_y = data_t.y 
            temp_x = data_t.drop(['y'], axis=1)
            temp_x = sm.add_constant(temp_x)


            quan_model_t = sm.OLS(temp_y, temp_x).fit()
            coe_est_t = quan_model_t.params
                
            coe_y_all=pd.concat([coe_y_all, coe_est_t], axis=1)
                
            ######## smooth estimation for DE ###################
        coe_y_all.columns=['t=%i' %i for i in range(self.m)]
            
        coe_y_smoothed = np.dot( np.array(coe_y_all), self.ker_mat_y ) / self.ker_mat_y.sum(axis=0).T
        coe_y_smoothed = pd.DataFrame(coe_y_smoothed, index=coe_y_all.index )  
            
            ######## errors of y ###############################           
        reg_mat_all = np.hstack( ( np.ones(self.nm).reshape(-1,1), self.vA , self.mS) )
            
        fitted_vY = None
        for t in range(self.m):
            fitted_y_tmp = reg_mat_all[ np.arange( t*self.n,(t+1)*self.n ) , :].dot( coe_y_smoothed.iloc[:, t].values )
            if fitted_vY is None:
                fitted_vY = fitted_y_tmp
            else:
                fitted_vY = np.hstack( (fitted_vY, fitted_y_tmp ))
                    
        resid_vY = self.vY - fitted_vY.reshape(-1,1) 
            
                
        coe_s_all={}
        coe_s_smoothed={}
        resid_mS =[]
        fitted_mS=None
            ###### pointwise estimation for DE ################
        for index_response in self.covariate_index:
                
            coe_est_S_index =pd.DataFrame()
                
            for t in np.delete(range(self.m),0):
                data_S_index_t ={'S_index': S_it[index_response][:, t], 'A': A_it[:, t-1]}
                
                reg_S_index_term = 'S_index~A'
                for index_covariate in self.covariate_index:
                    data_S_index_t[index_covariate]=S_it[index_covariate][:, t-1]
                    reg_S_index_term=reg_S_index_term + '+'+index_covariate
                
                data_S_index_t=pd.DataFrame(data_S_index_t)    
                temp_s_as_y = data_S_index_t.S_index
                temp_s_as_x = data_S_index_t.drop(['S_index'], axis=1)
                temp_s_as_x = sm.add_constant(temp_s_as_x)


                quan_model_S_index_t = sm.OLS(temp_s_as_y, temp_s_as_x).fit()
                coe_est_S_index_t = quan_model_S_index_t.params
                coe_est_S_index = pd.concat([coe_est_S_index, coe_est_S_index_t ], axis=1)
                   
            coe_s_all[index_response] = coe_est_S_index
            coe_est_S_smoothed_index = np.dot( np.array(coe_est_S_index), self.ker_mat_s) /self.ker_mat_s.sum(axis=0).T
            coe_est_S_smoothed_index = pd.DataFrame(coe_est_S_smoothed_index, index=coe_est_S_index.index)
                
            coe_est_S_smoothed_index.columns=['t=%i' %i for i in np.delete(range(self.m),self.m-1)]                  
            coe_s_smoothed[index_response] = coe_est_S_smoothed_index
                
                   
            ################ errors of S #######################               
            fitted_vS_index = None
            for tt in np.delete(range(self.m),self.m-1):
                fitted_vS_tmp = reg_mat_all[ np.arange( tt*self.n,(tt+1)*self.n ) , :].dot( coe_s_smoothed[index_response].loc[:, 't='+str(tt)].values ) ## original (tt-1)*self.n,tt*self.n
                if fitted_vS_index is None:
                   fitted_vS_index = fitted_vS_tmp
                else:
                   fitted_vS_index =np.hstack((fitted_vS_index, fitted_vS_tmp))
                
            if fitted_mS is None:
                fitted_mS = fitted_vS_index .reshape(-1,1)
            else:
                fitted_mS = np.hstack( (fitted_mS, fitted_vS_index .reshape(-1,1)  ))
                
        resid_mS = np.delete( self.mS, range(self.n), axis=0 ) - fitted_mS
        
        self.coe_y_smoothed = coe_y_smoothed
        self.coe_s_smoothed = coe_s_smoothed
        
        self.fitted_vY = fitted_vY.reshape(-1,1) 
        self.fitted_mS = fitted_mS
        self.reg_mat_all =reg_mat_all
                               
        return coe_y_smoothed, resid_vY, coe_s_smoothed, resid_mS
    
    def estimate_withoutA(self, y_it=None, S_it=None,  hc=None):  ### assume that no treatment effect
        if y_it is None: y_it = self. y_it
        if S_it is None: S_it = self.S_it
        if hc is None: hc=self.hc
            
            ###### pointwise estimation for DE ################
        coe_y_all=pd.DataFrame()
            
        for  t in range(self.m):
                
            #### dataframe for DE quantile regression ###
            data_t = {'y':y_it[:, t] }
                
            reg_term='y~'
            for index in self.covariate_index:
                data_t[index]=S_it[index][:, t]
                reg_term=reg_term + '+'+index
                
            data_t=pd.DataFrame(data_t)    
            temp_y = data_t.y 
            temp_x = data_t.drop(['y'], axis=1)
            temp_x = sm.add_constant(temp_x)


            quan_model_t = sm.OLS(temp_y, temp_x).fit()
            coe_est_t = quan_model_t.params
                
            coe_y_all=pd.concat([coe_y_all, coe_est_t], axis=1)
                
            ######## smooth estimation for DE ###################
        coe_y_all.columns=['t=%i' %i for i in range(self.m)]
            
        coe_y_smoothed = np.dot( np.array(coe_y_all), self.ker_mat_y ) / self.ker_mat_y.sum(axis=0).T
        coe_y_smoothed = pd.DataFrame(coe_y_smoothed, index=coe_y_all.index )  
            
            ######## errors of y ###############################           
        reg_mat_all = np.hstack( ( np.ones(self.nm).reshape(-1,1),  self.mS) )
            
        fitted_vY = None
        for t in range(self.m):
            fitted_y_tmp = reg_mat_all[ np.arange( t*self.n,(t+1)*self.n ) , :].dot( coe_y_smoothed.iloc[:, t].values )
            if fitted_vY is None:
                fitted_vY = fitted_y_tmp
            else:
                fitted_vY = np.hstack( (fitted_vY, fitted_y_tmp ))
                    
        resid_vY = self.vY - fitted_vY.reshape(-1,1) 
            
                
        coe_s_all={}
        coe_s_smoothed={}
        resid_mS =[]
        fitted_mS=None
            ###### pointwise estimation for DE ################
        for index_response in self.covariate_index:
                
            coe_est_S_index =pd.DataFrame()
                
            for t in np.delete(range(self.m),0):
                data_S_index_t ={'S_index': S_it[index_response][:, t]}
                
                reg_S_index_term = 'S_index~'
                for index_covariate in self.covariate_index:
                    data_S_index_t[index_covariate]=S_it[index_covariate][:, t-1]
                    reg_S_index_term=reg_S_index_term + '+'+index_covariate
                
                data_S_index_t=pd.DataFrame(data_S_index_t)    
                temp_s_as_y = data_S_index_t.S_index
                temp_s_as_x = data_S_index_t.drop(['S_index'], axis=1)
                temp_s_as_x = sm.add_constant(temp_s_as_x)


                quan_model_S_index_t = sm.OLS(temp_s_as_y, temp_s_as_x).fit()
                coe_est_S_index_t = quan_model_S_index_t.params
                coe_est_S_index = pd.concat([coe_est_S_index, coe_est_S_index_t ], axis=1)
                   
            coe_s_all[index_response] = coe_est_S_index
            coe_est_S_smoothed_index = np.dot( np.array(coe_est_S_index), self.ker_mat_s) /self.ker_mat_s.sum(axis=0).T
            coe_est_S_smoothed_index = pd.DataFrame(coe_est_S_smoothed_index, index=coe_est_S_index.index)
                
            coe_est_S_smoothed_index.columns=['t=%i' %i for i in np.delete(range(self.m),self.m-1)]                  
            coe_s_smoothed[index_response] = coe_est_S_smoothed_index
                
                   
            ################ errors of S #######################               
            fitted_vS_index = None
            for tt in np.delete(range(self.m),self.m-1):
                fitted_vS_tmp = reg_mat_all[ np.arange( tt*self.n,(tt+1)*self.n ) , :].dot( coe_s_smoothed[index_response].loc[:, 't='+str(tt)].values ) ## original (tt-1)*self.n,tt*self.n
                if fitted_vS_index is None:
                   fitted_vS_index = fitted_vS_tmp
                else:
                   fitted_vS_index =np.hstack((fitted_vS_index, fitted_vS_tmp))
                
            if fitted_mS is None:
                fitted_mS = fitted_vS_index .reshape(-1,1)
            else:
                fitted_mS = np.hstack( (fitted_mS, fitted_vS_index .reshape(-1,1)  ))
                
        resid_mS = np.delete( self.mS, range(self.n), axis=0 ) - fitted_mS
        
        self.coe_y_smoothed = coe_y_smoothed
        self.coe_s_smoothed = coe_s_smoothed
        
        self.fitted_vY = fitted_vY.reshape(-1,1) 
        self.fitted_mS = fitted_mS
        self.reg_mat_all =reg_mat_all
        
        coes_0={}
        coes_0['coe_y_smoothed_0']=coe_y_smoothed
        coes_0['resid_vY_0']=resid_vY
        coes_0['coe_s_smoothed_0']=coe_s_smoothed
        coes_0['resid_mS_0']=resid_mS
        coes_0['reg_mat_0']=reg_mat_all
                               
        return coes_0
            
    def QDE_QTE_calculate(self, time_start, time_end, gamma_y=None, Gamma_s=None, phi=None, beta_1=None):
        if gamma_y is None: gamma_y = self.coe_y_smoothed.loc['A'].values
        if beta_1 is None: beta_1 = self.coe_y_smoothed.drop(['const','A'], axis=0).values   ### d * m
                
        if Gamma_s is None: 
           Gamma_s={}
           phi={}
           for t in np.delete(range(self.m), self.m-1):
               Gamma_s['t='+str(t)] =None
               phi['t='+str(t)] = None
               for index_response in self.covariate_index:
                   if index_response==self.covariate_index[0]:
                       Gamma_s['t='+str(t)] = self.coe_s_smoothed[index_response].loc['A', 't='+str(t)].reshape(1, -1)
                       phi['t='+str(t)] =self.coe_s_smoothed[index_response].drop(['const','A']).loc[:, 't='+str(t)].values.reshape(1,self.d) 
                   else:
                      Gamma_s['t='+str(t)] = np.vstack( ( Gamma_s['t='+str(t)],  self.coe_s_smoothed[index_response].loc['A', 't='+str(t)].reshape(1, -1) ) )
                      phi['t='+str(t)] = np.vstack((phi['t='+str(t)], self.coe_s_smoothed[index_response].drop(['const','A']).loc[:, 't='+str(t)].values.reshape(1,self.d)   ))
                
        QDE=sum( gamma_y[np.arange(time_start,time_end)] )
                
        QIE=0
        for t in np.arange(max(1, time_start), time_end ):
            phi_Gamma=0
            for k in np.arange(0, t):  
                        
                phi_prod=np.identity(self.d)   
                        
                if k+1<=t-1:
                    for l in np.arange(k+1, t):   # from k+1 to t-1
                        phi_prod=phi['t='+str(l)].dot(phi_prod)   ### original phi_prod.dot(phi['t='+str(l)] ) 
                        
                phi_prod=phi_prod.dot( Gamma_s['t='+str(k)])
                        
                phi_Gamma = phi_Gamma + phi_prod
                        
            QIE = QIE + beta_1[:,t].reshape(1,self.d).dot(phi_Gamma)
                    
        return QDE, float(QIE)
                               
                
            
    def testing(self, B, time_start, time_end, hcb=None):
                
        Stat_QDE_b=[]
        Stat_QIE_b=[]
        Stat_QTE_b=[]
        
        if hcb is None: hcb=self.hcb
                
        coe_y_smoothed, resid_vY, coe_s_smoothed, resid_mS = self.estimate()
        QDE, QIE=self.QDE_QTE_calculate(time_start=time_start, time_end=time_end)

                
        for b in range(B):
                
            ##### generate state variablle ####
            reg_mat_b = copy.deepcopy(self.reg_mat_all )

            xi_y = np.random.normal(0, 1, self.n).reshape(-1, 1)

            xi_s = np.random.normal(0, 1, self.n*self.d).reshape(self.n, self.d)
            

            
            fitted_vY_b  = ( reg_mat_b[np.arange(0, self.n), :].dot( coe_y_smoothed.iloc[:, 0].values )).reshape(-1,1)  \
                        + xi_y*resid_vY[np.arange(0, self.n)]

            
            
            for t in np.delete( range(self.m),0):
                reg_s_diag_mat = reg_mat_b[np.arange(self.n*(t-1), t*self.n),:]              
                for xid in range(self.d): 
                    index = self.covariate_index[xid]                
                    reg_mat_b[np.arange(self.n*t, (t+1)*self.n), xid+2 ] =  ( reg_s_diag_mat.dot( coe_s_smoothed[index].loc[:, 't='+str(t-1)].values )) \
                        + xi_s[:, xid]*resid_mS[np.arange(self.n*(t-1), t*self.n), xid] 
                    
                fitted_vY_bt = ( reg_mat_b[np.arange(self.n*t, (t+1)*self.n), :].dot( coe_y_smoothed.iloc[:, t].values ) ).reshape(-1,1)  \
                        + xi_y*resid_vY[np.arange(self.n*t, (t+1)*self.n)]
                   
                fitted_vY_b = np.vstack( (fitted_vY_b, fitted_vY_bt))
                        
            S_it_b={}
            for xid in range(self.d): 
                S_it_b[self.covariate_index[xid] ] = ( reg_mat_b[:, xid+2]).reshape(self.n, self.m, order='F')

            #############################################
                        
            model_b = copy.deepcopy(self)
            model_b.y_it = fitted_vY_b.reshape(self.n, self.m, order='F')
            model_b.S_it = S_it_b
            
            coe_y_smoothed_b, resid_vY_b, coe_s_smoothed_b, resid_mS_b = model_b.estimate()
                                       
            QDE_b, QIE_b = model_b.QDE_QTE_calculate(time_start=time_start, time_end=time_end)
                    
            Stat_QDE_b.append( QDE_b - QDE)
            Stat_QIE_b.append( QIE_b - QIE )
            Stat_QTE_b.append( QIE_b +QDE_b- QIE-QDE )
                    

        
        mean_stat_QDE_b = np.mean(Stat_QDE_b)
        sd_stat_QDE_b = np.std( Stat_QDE_b )

        mean_stat_QIE_b = np.mean(Stat_QIE_b)
        sd_stat_QIE_b = np.std( Stat_QIE_b )
        
        QTE = QDE + QIE


        pvalue_QDE_twoSide = (  abs( np.array(Stat_QDE_b) ) > abs( QDE ) ).mean()
        pvalue_QIE_twoSide = (  abs( np.array(Stat_QIE_b) ) > abs( QIE ) ).mean()
        pvalue_QTE_twoSide = (  abs( np.array(Stat_QTE_b) ) > abs( QTE ) ).mean()
        
        pvalue_normal_QDE_twoSide = 2*(1 - norm.cdf( abs(QDE) /sd_stat_QDE_b  ) )
        
        pvalue_QDE_RightSide = (   np.array(Stat_QDE_b)  >  QDE  ).mean()
        pvalue_QIE_RightSide = (   np.array(Stat_QIE_b)  >  QIE  ).mean()
        pvalue_QTE_RightSide = (   np.array(Stat_QTE_b)  >  QTE  ).mean()
        
        pvalue_normal_QDE_RightSide = (1 - norm.cdf( QDE /sd_stat_QDE_b  ) )
        
        
        pvalues_all = np.array([[pvalue_QTE_twoSide, pvalue_QDE_twoSide ,pvalue_QIE_twoSide, pvalue_normal_QDE_twoSide ], [ pvalue_QTE_RightSide, pvalue_QDE_RightSide ,pvalue_QIE_RightSide, pvalue_normal_QDE_RightSide ] ])
        
        pvalues_all = pd.DataFrame(pvalues_all, columns=['QTE_boot', 'QDE_boot', 'QIE_boot', 'QDE_normal'], index=['Two_sides', 'Right_side'])
       
        return pvalues_all, mean_stat_QIE_b, sd_stat_QIE_b 





 