#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:27:53 2019

@author: davidblair
"""

import numpy as np
from scipy.stats import norm
from scipy.special import expit
from scipy.optimize import minimize

MAX_VAL=1e9
MAX_LOG = np.log(MAX_VAL)

class OrdinalRegression:
    
    def _transformBreakPoints(self,param_vec):
        param_vec[param_vec>MAX_LOG]=MAX_LOG
        return np.cumsum(np.exp(param_vec))
    
    def _untransformBreakPoints(self, param_vec):
        param_vec[1:]-=param_vec[:-1]
        return np.log(param_vec)
        
        
    def _tranformEffectParams(self,param_vec):
        param_vec[param_vec>MAX_LOG]=MAX_LOG
        return np.exp(param_vec)
        
        
    
    def _enforceSequential(self,data):
        allCats = set(list(data.ravel()))
        if -1 in allCats:
            allCats.remove(-1)
        
        if len(allCats)!=self.numClasses:
            assert len(allCats)==(self.numClasses-1), "Maximum of one entirely missing class allowed."
            missing_class=True
        else:
            missing_class=False
        value_range = set(range(min(allCats),max(allCats)+1))
        
        if missing_class:
            missing_val = value_range.difference(allCats)
            if len(missing_val)!=0:
                allCats.update(missing_val)
            else:
                if min(allCats)!=0:
                    #assumes missing category is lowest if no zero present
                    allCats.update([min(allCats)-1])
                else:
                    #if zero is present, then missing class is highest
                    allCats.update([max(allCats)+1])

        if allCats!=set(range(min(allCats),max(allCats)+1)):
            #re order so that they are  sequential
            old_cats =sorted(list(allCats))
            new_cats = range(0,len(allCats))
            cov_dict = dict(zip(old_cats,new_cats))
            for i in range(data.shape[0]):
                if data[i,0]!=-1:
                    data[i,0]=cov_dict[data[i,0]]
                
        
        if min(allCats)!=0:
            
            data[data[:,0]!=-1]-=min(allCats)
    
        return data
        
    def _generateBreakPoints(self,minVal=0.5,maxVal=2.5,alpha=2.0):
        b_k = np.random.beta(1.0,alpha,self.numClasses-1)
        
        break_points=np.zeros(self.numClasses-1)
        break_points[0]=b_k[0]
        break_points[1:]=b_k[1:]*np.cumprod(1.0-b_k[0:-1])
        return np.array([minVal+(maxVal-minVal)*np.cumsum(break_points)])
    
        
        
    def __init__(self,numClasses,linkFunction='Logit'):
        
        assert numClasses > 1, "Number of classes should be at least 2"
        
        self.numClasses=numClasses
        
        assert linkFunction in ['Logit','Probit'],"Only Logit and Probit link functions supported"
        
        if linkFunction=='Logit':
            self.linkFunction=lambda x:expit(x)
        else:
            self.linkFunction=lambda x:norm(0.0,1.0).cdf(x)
        
    def _categorical_sampler(self,p_vec):
        
        return np.random.choice(p_vec.shape[0], size=1, replace=True, p=p_vec)
    
    def _expectation(self,param_vec,covariates):
        prob_vals=np.zeros((covariates.shape[0],self.numClasses))
        _break_points=np.array([self._transformBreakPoints(param_vec[0:self.numClasses-1])])
        _effect_params = np.array([self._tranformEffectParams(param_vec[self.numClasses-1:])])
        latent_vals = np.sum(covariates*_effect_params,axis=1,keepdims=True)
        
        prob_vals=np.zeros((covariates.shape[0],self.numClasses))
        prob_vals[:,0:1] = self.linkFunction(_break_points[0,0]-latent_vals)
        prob_vals[:,1:-1] = self.linkFunction(_break_points[0,1:]-latent_vals)- self.linkFunction(_break_points[0,:-1]-latent_vals)
        prob_vals[:,-1:] = 1.0-self.linkFunction(_break_points[0,-1]-latent_vals)
        
        
        prob_vals[prob_vals>=1.0]=1.0-np.finfo(np.double).eps
        prob_vals[prob_vals<=-0.0]=np.finfo(np.double).eps
        
        return prob_vals/np.sum(prob_vals,axis=1,keepdims=True)

        
        
    def _neg_log_like_obs(self,param_vec,data,covariates,l2_penalty):
        
        
        _break_points=np.array([self._transformBreakPoints(param_vec[0:self.numClasses-1])])
        _effect_params = np.array([self._tranformEffectParams(param_vec[self.numClasses-1:])])
        
        latent_vals = np.sum(covariates*_effect_params,axis=1,keepdims=True)
        
        prob_vals=np.zeros((data.shape[0],self.numClasses))
        prob_vals[:,0:1] = self.linkFunction(_break_points[0,0]-latent_vals)
        prob_vals[:,1:-1] = self.linkFunction(_break_points[0,1:]-latent_vals)- self.linkFunction(_break_points[0,:-1]-latent_vals)
        prob_vals[:,-1:] = 1.0-self.linkFunction(_break_points[0,-1]-latent_vals)
        
        prob_vals[prob_vals>=1.0]=1.0-np.finfo(np.double).eps
        prob_vals[prob_vals<=-0.0]=np.finfo(np.double).eps
        
        log_prob_vals = np.log(prob_vals)
        return -1.0*np.sum(log_prob_vals[np.arange(data.shape[0]),data.ravel()])+l2_penalty*np.sum(_effect_params**2.0)
    
    def _neg_log_like_unobs(self,param_vec,obs_data,obs_cov,unobs_data,unobs_cov,l2_penalty):
        
        obs_neg_log_like=self._neg_log_like_obs(param_vec,obs_data,obs_cov,l2_penalty)
        
        _break_points=np.array([self._transformBreakPoints(param_vec[0:self.numClasses-1])])
        _effect_params = np.array([self._tranformEffectParams(param_vec[self.numClasses-1:])])
        
        latent_vals = np.sum(unobs_cov*_effect_params,axis=1,keepdims=True)
        
        prob_vals=np.zeros((unobs_data.shape[0],self.numClasses))
        prob_vals[:,0:1] = self.linkFunction(_break_points[0,0]-latent_vals)
        prob_vals[:,1:-1] = self.linkFunction(_break_points[0,1:]-latent_vals)- self.linkFunction(_break_points[0,:-1]-latent_vals)
        prob_vals[:,-1:] = 1.0-self.linkFunction(_break_points[0,-1]-latent_vals)
        
        prob_vals[prob_vals>=1.0]=1.0-np.finfo(np.double).eps
        prob_vals[prob_vals<=-0.0]=np.finfo(np.double).eps
        
        log_prob_vals = np.log(prob_vals)
        return np.sum(-1.0*log_prob_vals*unobs_data)+obs_neg_log_like
        
        
        
        
    def FitModel(self,data,covariates,tol=1e-6,max_bfgs_iter=2000,max_em_steps=200,l2_penalty=1.0):
        if len(data.shape)<2:
            data=data.reshape(data.shape[0],1)
        data=self._enforceSequential(data)
        
        
        assert len(covariates.shape)==2,"Expects a 2-d array of coviariates"
        assert covariates.shape[0]==data.shape[0],"Dimensions of covariates and data must match"
        
        param_vec = np.zeros((self.numClasses-1)+covariates.shape[1])
        param_vec[:self.numClasses-1] = self._untransformBreakPoints(self._generateBreakPoints())
        
        param_vec[self.numClasses-1:]=0.0
        
        if np.sum(data==-1)==0:
            max_like_output=minimize(self._neg_log_like_obs,param_vec,args=(data,covariates,l2_penalty),tol=tol,options={'maxiter':max_bfgs_iter})
            param_vec=max_like_output['x']
            _break_points=self._transformBreakPoints(param_vec[0:self.numClasses-1])
            _effect_params = self._tranformEffectParams(param_vec[self.numClasses-1:])
            
            
            return {'log-like':-1.0*max_like_output['fun'],'break_points':_break_points,'effect_params':_effect_params,'predicted_labels':None}
        else:
            
            where_missing = np.where(data==-1)[0]
            where_not_missing = np.where(data!=-1)[0]
            
            obs_data=data[where_not_missing]
            obs_cov = covariates[where_not_missing]
            
            unobs_data = np.zeros((where_missing.shape[0],self.numClasses))
            unobs_cov = covariates[where_missing]
            
            
            unobs_data[:,:]=self._expectation(param_vec,unobs_cov)
            
            exp_joint_like = self._neg_log_like_unobs(param_vec,obs_data,obs_cov,unobs_data,unobs_cov,l2_penalty)
            prev_loss = exp_joint_like+np.sum(unobs_data*np.log(unobs_data))
            for i in range(max_em_steps):
                
                #M-step
                max_like_output=minimize(self._neg_log_like_unobs,param_vec,args=(obs_data,obs_cov,unobs_data,unobs_cov,l2_penalty),tol=tol,options={'maxiter':max_bfgs_iter})
                param_vec=max_like_output['x']
                
                #E-step
                unobs_data[:,:]=self._expectation(param_vec,unobs_cov)
                
                #compute loss
                exp_joint_like = self._neg_log_like_unobs(param_vec,obs_data,obs_cov,unobs_data,unobs_cov,l2_penalty)
                loss = exp_joint_like+np.sum(unobs_data*np.log(unobs_data))
                error=(prev_loss-loss)/loss
                if error < tol:
                    break
                else:
                    prev_loss=loss
                    
            if (i+1)==max_em_steps:
                print("Warning: EM algorithm did not converge in {} iterations.".format(max_em_steps))
                
            _break_points=self._transformBreakPoints(param_vec[0:self.numClasses-1])
            _effect_params = self._tranformEffectParams(param_vec[self.numClasses-1:])
            
            return {'log-like':-1.0*loss,'break_points':_break_points,'effect_params':_effect_params,'predicted_labels':dict(zip(where_missing,unobs_data))}
    
        
    def SimulateData(self,numSamples,numCovariates, **kwargs):
        
        allKeywordArgs = list(kwargs.keys())
        
        if 'breakPointRange' not in allKeywordArgs:
            breakPointRange=[0.5,2.5]
        else:
            breakPointRange=kwargs['breakPointRange']
        
        if 'breakPointAlpha' not in allKeywordArgs:
            breakPointAlpha=1.0
        else:
            breakPointAlpha=kwargs['breakPointAlpha']
            
        if 'effectParamPriors' not in allKeywordArgs:
            effectParamPrior=1.0
        else:
            effectParamPrior=kwargs['effectParamPrior']
            
        
        
        covariates = np.random.gamma(1.0,1.0,size=(numSamples,numCovariates))
        break_points = self._generateBreakPoints(minVal=breakPointRange[0],maxVal=breakPointRange[1],alpha=breakPointAlpha)
        effect_params = np.random.exponential(effectParamPrior,numCovariates)
        latent_vals = np.sum(covariates*effect_params,axis=1,keepdims=True)
        
        prob_vals=np.zeros((numSamples,self.numClasses))
        
        prob_vals[:,0:1] = self.linkFunction(break_points[0,0]-latent_vals)
        prob_vals[:,1:-1] = self.linkFunction(break_points[0,1:]-latent_vals)- self.linkFunction(break_points[0,:-1]-latent_vals)
        prob_vals[:,-1:] = 1.0-self.linkFunction(break_points[0,-1]-latent_vals)
        
        obs_vals=np.apply_along_axis(self._categorical_sampler,1,prob_vals)
        return {'obs_vals':obs_vals,'effect_params':effect_params,'covariates':covariates,'break_points':break_points}
        
        
        
if __name__=='__main__':
    test=OrdinalRegression(3)
    o=test.SimulateData(200,3)
    p_1=test.FitModel(o['obs_vals'],o['covariates'])
    o['obs_vals'][np.random.binomial(1,0.4,o['obs_vals'].shape[0])==1]=-1
    p_2=test.FitModel(o['obs_vals'],o['covariates'])
#    