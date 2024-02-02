import numpy as np
import scipy as sp
from scipy.stats import norm
import datetime
import multiprocessing as mp


import pandas as pd
import statsmodels.formula.api as smf
import copy
import scipy.linalg
from random import choices

import statsmodels.api as sm




import time

##############################################
def ker(t, h):
    return np.exp(-t**2/h**2)

class SVCM_without_barA(object):
    def __init__(self, y_rit, A_rit, S_rit, time_points, tau, ker_mat_r, hc, hcb):
        
        ## y_rit: R*n*m  array
        ## A_rit: R*n*m  array
        ## S_rit: dictionary {'S1': S_rit1, ..., 'Sd': S_ritd}, each S_itj is an R*n*m array
        ## time_points: t_j 
        ## hc: bandwidth for the coefficients
        ## tau: quantile level
        
        # main args
        self.y_rit = y_rit
        self.S_rit = S_rit
        self.A_rit = A_rit

        self.d = len(S_rit)
        self.tau=tau
        self.hcb=hcb
         
        self.R,self.n,self.m = y_rit.shape
        self.nm =self.n*self.m
        self.time_points = time_points 
        self.hc = hc
      
        self.covariate_index=[key for key in  S_rit.keys()]
                
                
        ### generate kernel matrix grids ######
        self.ker_mat_r = ker_mat_r
        
        if self.hc==0:
            self.ker_mat_y = np.identity(self.m)  ## the orginal estimator without smoothing
            self.ker_mat_s = np.identity(self.m -1)
        else:
            self.ker_mat_y = ker( np.array(self.time_points ).reshape(-1,1) - np.array(self.time_points ).reshape(1,-1), self.hc)
            time_points_s = np.delete(self.time_points, self.m-1)
            #self.ker_mat_s = ker( np.array(time_points_s ).reshape(-1,1) - np.array(time_points_s ).reshape(1,-1), self.hc)
            self.ker_mat_s = np.identity(self.m -1)
       
        #################
        self.reg_mat_all=None
        self.vY=None
        self.mS=None        
             
        ### estimates ###
        self.coe_y_smoothed = None
        self.coe_y_smoothed_0 = None
        self.coe_s_smoothed = None
        self.coe_s_smoothed_0 =None
        
            
        self.coe_y_smoothed_region=None
        self.coe_s_smoothed_region=None
        
        self.fitted_vY = None
        self.fitted_mS = None
        
        self.resid_vY=None
        self.resid_mS=None
        
        self.QDE=None
        self.QIE=None
        
    def estimate_within_region(self, r, y_it=None, S_it=None,  A_it=None, hc=None):
        if y_it is None: y_it = self. y_rit[r]
        if S_it is None: 
            S_it = {}
            for key, value in  self.S_rit.items():
                S_it[key] = value[r]
        if A_it is None: A_it = self.A_rit[r]
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
            

        ########################################    
                
        coe_s_all={}
        coe_s_smoothed={}
        #resid_mS =[]
        #fitted_mS=None
            ###### pointwise estimation for DE ################
        for index_response in self.covariate_index:
                
            coe_est_S_index =pd.DataFrame()
                
            for t in np.delete(range(self.m),0):
                data_S_index_t ={'S_index': S_it[index_response][:, t], 'A': A_it[:, t-1] }
                
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
            coe_s_smoothed[index_response]= coe_est_S_smoothed_index 
                                                                         
        return coe_y_smoothed, coe_s_smoothed
    
    def estimate_allRegions(self):
        
        ####### reg_mat_all for regression #####################################
        self.reg_mat_all={}
        self.vY={}
        self.mS={}
        
        self.coe_y_smoothed=[]
        self.coe_s_smoothed={}
        
        for index in self.covariate_index:
            self.coe_s_smoothed[index]=[]
        
        for r in range(self.R):
            
            vA_r = self.A_rit[r].reshape(-1,1, order='F')
            
            coe_y_smoothed_r, coe_s_smoothed_r=self.estimate_within_region(r=r)
            self.coe_y_smoothed.append( np.array(coe_y_smoothed_r ) )
            
            mS_r =None       
            for index, value in self.S_rit.items():
                
                self.coe_s_smoothed[index].append( np.array( coe_s_smoothed_r[index]) )
                
                if mS_r is None:
                    mS_r = value[r].reshape(-1,1, order='F')
                else:
                    mS_r = np.hstack( (mS_r, value[r].reshape(-1,1, order='F') ) )
                    

            self.reg_mat_all['r='+str(r)]= np.hstack( ( np.ones(self.nm).reshape(-1,1), vA_r ,  mS_r) )
            self.vY['r='+str(r)] = self.y_rit[r].reshape(-1,1, order='F')
            self.mS['r='+str(r)] =   mS_r
        
        ############################################
        
        coe_y_smoothed_3D = np.array(self.coe_y_smoothed)   # R*p*m
              
        ########### smoothing regions #######################
        coe_y_smoothed_region = ( coe_y_smoothed_3D.T.dot( self.ker_mat_r ) ).T
        
        coe_s_smoothed_region = {}
        
        for index_response in self.covariate_index:
             coe_s_smoothed_3D = np.array( self.coe_s_smoothed[index_response] )   # R*p*(m-1)
             coe_s_smoothed_region[index_response] = (coe_s_smoothed_3D.T.dot(self.ker_mat_r) ).T
            
        ######## errors of y ###############################
        resid_vY={}
        resid_mS={}
        
        self.fitted_vY ={}
        self.fitted_mS={}

        ####################################################          
        for r in range(self.R):
            reg_mat_r = self.reg_mat_all['r='+str(r)]
            
            fitted_vY_r = []
            fitted_mS_index_r=[]
            for tt in np.delete(range(self.m),self.m-1):
                fitted_vY_r.append(  reg_mat_r[ np.arange( tt*self.n,(tt+1)*self.n) , :].dot( coe_y_smoothed_region[r, :, tt] )  )
                
                fitted_vS_index=[]
                for index_response in self.covariate_index: 
                    fitted_vS_tmp = reg_mat_r[ np.arange( tt*self.n,(tt+1)*self.n ) , :].dot( coe_s_smoothed_region[index_response][r, :, tt] ) ## original (tt-1)*self.n,tt*self.n
                    fitted_vS_index.append( fitted_vS_tmp   )
                
                fitted_vS_index = np.vstack(fitted_vS_index).T  ## n*d
                fitted_mS_index_r.append(fitted_vS_index )
                
            fitted_mS_r = np.vstack(fitted_mS_index_r) 
            fitted_vY_r.append( reg_mat_r[ np.arange( (self.m-1)*self.n,self.m*self.n ) , :].dot( coe_y_smoothed_region[r, :, self.m-1] ) )     
            resid_mS_r = np.delete( self.mS['r='+str(r)], range(self.n), axis=0 ) - fitted_mS_r
            
            self.fitted_vY['r='+str(r)] = np.hstack( fitted_vY_r).reshape(self.nm, 1)
            self.fitted_mS['r='+str(r)]  = fitted_mS_r
            
            resid_vY['r='+str(r)] = self.vY['r='+str(r)] -  self.fitted_vY['r='+str(r)]
            resid_mS['r='+str(r)] = resid_mS_r
            
        self.coe_y_smoothed_region=coe_y_smoothed_region
        self.coe_s_smoothed_region=coe_s_smoothed_region
        self.resid_vY=resid_vY
        self.resid_mS=resid_mS
            
        return coe_y_smoothed_region, coe_s_smoothed_region, resid_vY, resid_mS

    
    
    def estimate_within_region_withoutA(self, r, y_it=None, S_it=None, hc=None):
        if y_it is None: y_it = self. y_rit[r]
        if S_it is None: 
            S_it = {}
            for key, value in  self.S_rit.items():
                S_it[key] = value[r]
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
            

        ########################################    
                
        coe_s_all={}
        coe_s_smoothed={}
        #resid_mS =[]
        #fitted_mS=None
            ###### pointwise estimation for DE ################
        for index_response in self.covariate_index:
                
            coe_est_S_index =pd.DataFrame()
                
            for t in np.delete(range(self.m),0):
                data_S_index_t ={'S_index': S_it[index_response][:, t] }
                
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
            coe_s_smoothed[index_response]= coe_est_S_smoothed_index 
                                              
                           
        return coe_y_smoothed, coe_s_smoothed
    
    def estimate_allRegions_withoutA(self):
        
        ####### reg_mat_all for regression #####################################
            
        self.reg_mat_all={}
        self.vY={}
        self.mS={}
        
        self.coe_y_smoothed_0=[]
        self.coe_s_smoothed_0={}
        
        for index in self.covariate_index:
            self.coe_s_smoothed_0[index]=[]
        
        for r in range(self.R):
                       
            coe_y_smoothed_r, coe_s_smoothed_r=self.estimate_within_region_withoutA(r=r)
            self.coe_y_smoothed_0.append( np.array(coe_y_smoothed_r ) )
            
            mS_r =None       
            for index, value in self.S_rit.items():
                
                self.coe_s_smoothed_0[index].append( np.array(coe_s_smoothed_r[index] ) )
                
                if mS_r is None:
                    mS_r = value[r].reshape(-1,1, order='F')
                else:
                    mS_r = np.hstack( (mS_r, value[r].reshape(-1,1, order='F') ) )
                    

            self.reg_mat_all['r='+str(r)]= np.hstack( ( np.ones(self.nm).reshape(-1,1),  mS_r) )
            self.vY['r='+str(r)] = self.y_rit[r].reshape(-1,1, order='F')
            self.mS['r='+str(r)] =   mS_r            
        
        ############################################
        
        coe_y_smoothed_3D = np.array(self.coe_y_smoothed_0)   # R*p*m
              
        ########### smoothing regions #######################
        coe_y_smoothed_region = ( coe_y_smoothed_3D.T.dot( self.ker_mat_r ) ).T
        
        coe_s_smoothed_region = {}
        
        for index_response in self.covariate_index:
             coe_s_smoothed_3D = np.array( self.coe_s_smoothed_0[index_response] )   # R*p*(m-1)
             coe_s_smoothed_region[index_response] = (coe_s_smoothed_3D.T.dot(self.ker_mat_r) ).T
            
        ######## errors of y ###############################
        resid_vY={}
        resid_mS={}

        self.fitted_vY ={}
        self.fitted_mS={}
        ####################################################          
        for r in range(self.R):
            reg_mat_r = self.reg_mat_all['r='+str(r)]
            
            fitted_vY_r = []
            fitted_mS_index_r=[]
            for tt in np.delete(range(self.m),self.m-1):
                fitted_vY_r.append(  reg_mat_r[ np.arange( tt*self.n,(tt+1)*self.n) , :].dot( coe_y_smoothed_region[r, :, tt] )  )
                
                fitted_vS_index=[]
                for index_response in self.covariate_index: 
                    fitted_vS_tmp = reg_mat_r[ np.arange( tt*self.n,(tt+1)*self.n ) , :].dot( coe_s_smoothed_region[index_response][r,:, tt] ) ## original (tt-1)*self.n,tt*self.n
                    fitted_vS_index.append( fitted_vS_tmp   )
                
                fitted_vS_index = np.vstack(fitted_vS_index).T  # n*d
                fitted_mS_index_r.append(fitted_vS_index )
                
            fitted_mS_r = np.vstack(fitted_mS_index_r) 
            fitted_vY_r.append( reg_mat_r[ np.arange( (self.m-1)*self.n,self.nm ) , :].dot( coe_y_smoothed_region[r, :, self.m-1] ) )     
            resid_mS_r = np.delete( self.mS['r='+str(r)], range(self.n), axis=0 ) - fitted_mS_r
            
            self.fitted_vY['r='+str(r)] = np.hstack( fitted_vY_r).reshape(self.nm, 1)
            self.fitted_mS['r='+str(r)]  = fitted_mS_r
            
            resid_vY['r='+str(r)] = self.vY['r='+str(r)] -  self.fitted_vY['r='+str(r)]
            resid_mS['r='+str(r)] = resid_mS_r
            
        coes_0={}
        coes_0['coe_y_smoothed_0']=coe_y_smoothed_region
        coes_0['resid_vY_0']=resid_vY
        coes_0['coe_s_smoothed_0']=coe_s_smoothed_region
        coes_0['resid_mS_0']=resid_mS
        coes_0['reg_mat_0']=self.reg_mat_all
                               
        return coes_0            
            
    def QDE_QTE_oneRegion(self, time_start, time_end, r=None, gamma_y=None, Gamma_s=None, phi=None, beta_1=None):
        if gamma_y is None: gamma_y = self.coe_y_smoothed_region[r,1,:]
        if beta_1 is None: beta_1 = np.delete(self.coe_y_smoothed_region[r], [0,1], axis=0)   ### d * m
                
        if Gamma_s is None: 
           Gamma_s={}
           phi={}
           for t in np.delete(range(self.m), self.m-1):
               Gamma_s['t='+str(t)] =None
               phi['t='+str(t)] = None
               for index_response in self.covariate_index:
                   if index_response==self.covariate_index[0]:
                       Gamma_s['t='+str(t)] = self.coe_s_smoothed_region[index_response][r,1,t].reshape(1, -1)
                       phi['t='+str(t)] =np.delete( self.coe_s_smoothed_region[index_response][r], [0,1], axis=0 )[:, t].reshape(1,self.d) 
        
                   else:
                      Gamma_s['t='+str(t)] = np.vstack( ( Gamma_s['t='+str(t)],  self.coe_s_smoothed_region[index_response][r,1,t].reshape(1, -1) ) )
                      phi['t='+str(t)] = np.vstack((phi['t='+str(t)], np.delete( self.coe_s_smoothed_region[index_response][r], [0,1], axis=0 )[:, t].reshape(1,self.d)    ))
                
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
    
    def QDE_QTE_AllRegion(self, time_start, time_end):
        
        QDE_all_region=[]
        QIE_all_region=[]
        
        for r in range(self.R):
            QDE,QIE=self.QDE_QTE_oneRegion(time_start=time_start, time_end=time_end, r=r)
            QDE_all_region.append(QDE)
            QIE_all_region.append(QIE)  
            
        QDE_all = sum(QDE_all_region)
        QIE_all = sum(QIE_all_region)
        
        self.QDE=QDE_all
        self.QIE=QIE_all
        
        return QDE_all, QIE_all
                              
            
    def testing(self, B, time_start, time_end, hcb=None):
                
        Stat_QDE_b=[]
        Stat_QIE_b=[]
        Stat_QTE_b=[]
        
        if hcb is None: hcb=self.hcb
        
        coe_y_smoothed_region = self.coe_y_smoothed_region
        coe_s_smoothed_region = self.coe_s_smoothed_region
        resid_vY = self.resid_vY
        resid_mS = self.resid_mS
        
        QDE=self.QDE
        QIE=self.QIE
        
        if QDE is None:
           coe_y_smoothed_region, coe_s_smoothed_region, resid_vY, resid_mS = self.estimate_allRegions()
           QDE, QIE=self.QDE_QTE_AllRegion(time_start=time_start, time_end=time_end)
       

       ############# center the residuals ##########################
        resid_s_matrix=[]
        
        resid_vY_matrix = np.array(  [resid_vY['r='+str(r)].reshape(self.n, self.m, order='F') for r in range(self.R)] ) # R*n*m
        resid_vY_matrix = resid_vY_matrix - np.quantile(resid_vY_matrix,  q=self.tau, axis=1 ).reshape(self.R, 1, self.m)
        
        
        for xid in range(self.d):      
            resid_s_index_matrix_r = np.array([ resid_mS['r='+str(r)][:, xid].reshape(self.n, self.m-1, order='F' ) for r in range(self.R)]  ) # R*n*(m-1)
            resid_s_index_matrix_r = resid_s_index_matrix_r -  np.quantile(resid_s_index_matrix_r,  q=self.tau, axis=1 ).reshape(self.R, 1, self.m-1)   
            resid_s_matrix.append(resid_s_index_matrix_r)
        
      #################################################################
                 
        for b in range(B):
                
            ##### generate errors ####
            reg_mat_b = copy.deepcopy(self.reg_mat_all )
            
            xi_y = np.random.normal(0, 1, self.n).reshape(-1, 1)

            xi_s = np.random.normal(0, 1, self.n*self.d).reshape(self.n, self.d)

            #positions = choices( list(range(self.n)),  k=self.n )

            resid_vY_matrix_b = resid_vY_matrix
            
            #positions_s = [choices( list(range(self.n)),  k=self.n ) for _ in range(self.d)]
            
            resid_mS_matrix_b =[]
            S_rit_b={}
            for xid in range(self.d):
                resid_vS_tmp = resid_s_matrix[ xid ]
                resid_mS_matrix_b.append( resid_vS_tmp )
                S_rit_b[self.covariate_index[xid] ] =[]
            
            #######################################
        ####################################################   
            Y_rit_b=[]
            for r in range(self.R):
                reg_mat_b_r = reg_mat_b['r='+str(r)]
                
                #resid_vY_b_r = resid_vY_matrix_b[r].reshape(-1,1, order='F')
                fitted_vY_b_r  = ( reg_mat_b_r[np.arange(0, self.n), :].dot( coe_y_smoothed_region[r][:, 0] )).reshape(-1,1)  \
                        + xi_y*resid_vY_matrix_b[r][:,0].reshape(-1,1)                
                
                resid_mS_matrix_b_r =[ resid_mS_matrix_b[xid][r] for xid in range(self.d)  ] 

                ####################################    
                for t in np.delete( range(self.m),0):
                    reg_s_diag_mat_r = reg_mat_b_r[np.arange(self.n*(t-1), t*self.n),:]              
                    for xid in range(self.d): 
                        index = self.covariate_index[xid]                
                        reg_mat_b_r[np.arange(self.n*t, (t+1)*self.n), xid+2 ] =  ( reg_s_diag_mat_r.dot( coe_s_smoothed_region[index][r][:, t-1] )) \
                            + xi_s[:, xid]*resid_mS_matrix_b_r[xid][:,t-1]
                        
                    fitted_vY_bt_r = ( reg_mat_b_r[np.arange(self.n*t, (t+1)*self.n), :].dot( coe_y_smoothed_region[r][:, t] ) ).reshape(-1,1)  \
                            +  xi_y*resid_vY_matrix_b[r][:,t].reshape(-1,1)  
                       
                    fitted_vY_b_r = np.vstack( (fitted_vY_b_r, fitted_vY_bt_r))
                    
                Y_rit_b.append( fitted_vY_b_r.reshape(self.n, self.m, order='F') )
                            
                for xid in range(self.d): 
                    S_rit_b[self.covariate_index[xid] ].append( ( reg_mat_b_r[:, xid+2]).reshape(self.n, self.m, order='F') )
                    
            for xid in range(self.d):
                S_rit_b[self.covariate_index[xid] ] =np.array(S_rit_b[self.covariate_index[xid] ])  ## each S R*n*m
            

            #############################################
                        
            model_b = copy.deepcopy(self)
            model_b.y_rit = np.array(Y_rit_b)
            model_b.S_rit = S_rit_b
                    
            #############################################
            coe_y_smoothed_region_b, coe_s_smoothed_region_b, resid_vY_b, resid_mS_b = model_b.estimate_allRegions()
            QDE_b, QIE_b=model_b.QDE_QTE_AllRegion(time_start=time_start, time_end=time_end)
                    
            Stat_QDE_b.append( QDE_b - QDE)
            Stat_QIE_b.append( QIE_b - QIE )
            Stat_QTE_b.append( QIE_b +QDE_b- QIE-QDE )
                    
        #np.savetxt('./Boot1/boot'+str(int(np.random.randn(1)[0]*10))+'.txt', np.vstack( ( Stat_QDE_b, np.repeat(QDE, B),
                                                                                       # Stat_QIE_b, np.repeat(QIE, B) ) ) )
        
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


def generate_temporalData(n, m, TI,  sigma_eta, tau, delta_eta, delta_DE, delta_IE):
    # n: number of dates
    # m: length of time
    # TI: time span for treatment
    # rho: correlation for day-specific covariance AR(rho)
    # delta_eta: variance of eta
    # delta_DE: signal stregth of DE
    # delta_IE: signal_stregth of IE
    
    ##### time sclaed to 1 ###########
    tm = np.array( range(m) )/m
    
    ##### True coefficients ##########
    phi_01 = tm*2*(1 + tau /2)
    phi_02 = tm*(1 + tau /3)
    
    phi_11 = np.sin( range(m) )*0.3*( 1 + tau/3 )
    phi_12 = np.sin( range(m) )*0.5*( 1 + tau/4 )
    phi_21 = np.cos( range(m) )*0.3*( 1 + tau/3 )
    phi_22 = np.cos( range(m) )*0.5*( 1 + tau/4 )
    
    Gamma_s1 = tm**2*tau*delta_IE
    Gamma_s2 = tm**3*tau*delta_IE
    
    beta_0 = tm*3*(1 + tau)
    beta_0_all = np.array( np.tile(beta_0, n)).reshape(n,m)
    
    beta_11 = tm*tau+0.3
    beta_11_all = np.array( np.tile(beta_11, n)).reshape(n,m)
    
    beta_12 =  np.sin(tm)*tau + 0.4 
    beta_12_all = np.array( np.tile(beta_12, n)).reshape(n,m)
    
    gamma_y = np.cos(tm)*(1 + 2*tau)*0.1*delta_DE
    gamma_y_all = np.array( np.tile(gamma_y, n)).reshape(n,m)
    
    ##### generate errors #######
    sigma_eta=sigma_eta*delta_eta
    
    Sigma_y = sigma_eta + 0.3*np.identity(m)
    Sigma_s1 = sigma_eta + 0.2*np.identity(m)
    Sigma_s2 = sigma_eta + 0.2*np.identity(m)
    
    error_y = np.random.multivariate_normal( np.repeat(0,m), Sigma_y, n ) - norm.ppf(tau, 0, np.sqrt(delta_eta+0.3) )
    error_s1 = np.random.multivariate_normal( np.repeat(0,m), Sigma_s1, n ) - norm.ppf(tau, 0, np.sqrt(delta_eta+0.2) )
    error_s2 = np.random.multivariate_normal( np.repeat(0,m), Sigma_s2, n ) - norm.ppf(tau, 0, np.sqrt(delta_eta+0.2) )
    
    #### generate treatment #####  
    AB_vector = np.tile( np.repeat( [1, 0], TI ), m// TI //2 )
    BA_vector = np.tile( np.repeat( [0, 1], TI ), m// TI// 2 ) 
    vec =np.hstack([AB_vector, BA_vector])
    
    A_it =np.array( np.tile(vec, n//2) ).reshape(n, m)
    
    
    ##### generate state variablle ####
    S1_t0 = np.random.normal(0, 1, n)
    S2_t0 = np.random.normal(0, 1, n)
    
    S1_t = S1_t0
    S2_t = S2_t0
    
    S1_it = S1_t0
    S2_it = S2_t0
    
    for t in np.delete( range(m),0):
        S1_ittmp = phi_01[t-1] + phi_11[t-1]*S1_t  +  phi_12[t-1]*S2_t + A_it[:, t-1]*Gamma_s1[t-1]+error_s1[:,t]
        S2_ittmp = phi_02[t-1] + phi_21[t-1]*S1_t  +  phi_22[t-1]*S2_t + A_it[:, t-1]*Gamma_s2[t-1]+error_s2[:,t]
        
        S1_it = np.vstack( [S1_it, S1_ittmp] )
        S1_t = S1_ittmp
        
        S2_it = np.vstack( [S2_it, S2_ittmp] )
        S2_t  = S2_ittmp
    
    S1_it=S1_it.T
    S2_it=S2_it.T
        
    y_it = beta_0_all + S1_it*beta_11_all + S2_it*beta_12_all + A_it*gamma_y_all + error_y
    
    final_S_it={}
    final_S_it['S1']=S1_it
    final_S_it['S2']=S2_it
    
    
    return y_it, final_S_it, A_it 





 