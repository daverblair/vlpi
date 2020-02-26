#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:44:37 2020

@author: davidblair
"""

import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize

MAX_VAL=1e9
MAX_LOG = np.log(MAX_VAL)

class MonotonicBinaryClassifier:
    def __init__(self):
        self.linkFunction=lambda x:expit(x)
        self.offset = None
        self.effect_params=None
            
            
    def _neg_log_like(self,param_vec,data,covariates,l1_penalty):
        
        base_vals = param_vec[0]+np.sum(param_vec[1:]*covariates,axis=1)
        prob_vals = self.linkFunction(base_vals)
        prob_vals[prob_vals>=1.0]=1.0-np.finfo(np.double).eps
        prob_vals[prob_vals<=0.0]=np.finfo(np.double).eps
        
        return -1.0*(np.sum(data.ravel()*np.log(prob_vals))+np.sum((1.0-data.ravel())*np.log(1.0-prob_vals)))+l1_penalty*np.sum(np.abs(param_vec[1:]))

            
    def FitModel(self,data,covariates,tol=1e-6,max_bfgs_iter=2000,l1_penalty=1.0):
        if data.dtype!=np.int:
            data=np.array(data,dtype=np.int)
            
        if len(data.shape)<2:
            data=data.reshape(data.shape[0],1)
        
        
        assert len(covariates.shape)==2,"Expects a 2-d array of coviariates"
        assert covariates.shape[0]==data.shape[0],"Dimensions of covariates and data must match"
        
        init_param_vec = np.zeros(1+covariates.shape[1])
        
        init_param_vec[0] = logit(np.sum(data)/data.shape[0])
        init_param_vec[1:]=np.ones(covariates.shape[1])*1e-5
        bounds = [(-np.inf,np.inf)]+[(0.0,np.inf) for x in range(covariates.shape[1])]
        max_like_output=minimize(self._neg_log_like,init_param_vec,args=(data,covariates,l1_penalty),tol=tol,bounds=bounds,options={'maxiter':max_bfgs_iter})
        self.offset = max_like_output['x'][0]
        self.effect_params = max_like_output['x'][1:]
        
        return {'log-like':-1.0*max_like_output['fun'],'threshold':self.offset,'effect_params':self.effect_params}
    
    def ComputeLogOdds(self,covariates):
        assert self.offset is not None,'Model has not been provided data for fitting'
        return self.offset+np.sum(self.effect_params*covariates,axis=1)
    
    

if __name__=='__main__':

    from vlpi.data.ClinicalDataSimulator import ClinicalDataSimulator
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    import torch
    

    numSamples = 10000
    numAssociatedTraits=20
    nLatentSimDim=4
    simulateLabels=True

    numCovPerClass = [2,3,10]
    covNames = ['A','B','C']


    simulator = ClinicalDataSimulator(numAssociatedTraits,nLatentSimDim,numCovPerClass,interceptPriors=[-3.0,1.5])
    simData=simulator.GenerateClinicalData(numSamples)
    labelData=simulator.GenerateTargetDx(simData['latent_phenotypes'])


    log_reg=MonotonicBinaryClassifier()
    output=log_reg.FitModel(labelData['target_dx_data'].numpy(),simData['latent_phenotypes'].numpy(),tol=1e-8)
    pred = log_reg.ComputeLogOdds(simData['latent_phenotypes'].numpy())
    prec_recall = precision_recall_curve(labelData['target_dx_data'].numpy().ravel(),pred)
    prec_recall_ideal = precision_recall_curve(labelData['target_dx_data'].numpy().ravel(),torch.sum(simData['latent_phenotypes']*labelData['model_params']['targetDxMap'],axis=1).numpy())
    plt.step(prec_recall[1],prec_recall[0],'r')
    plt.step(prec_recall_ideal[1],prec_recall_ideal[0],'b')   
        
        
        