#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:21:44 2019

@author: davidblair
"""
import torch
import pyro
import pyro.distributions as dist
from vlpi.utils import random_catcov,infer_liability_CI
from sklearn.datasets import make_sparse_spd_matrix
import numpy as np

class ClinicalDataSimulator:
    
    def _gen_rand_cov_mat(self,dim,sparsity):
        C=np.abs(make_sparse_spd_matrix(dim,norm_diag=False,alpha=sparsity,smallest_coef=0.1,largest_coef=1.0))
        return torch.tensor(C/np.diag(C).max(),dtype=torch.float32)
        
    
    def _folded_multivariate_TDist_RVs(self,C,df,sample_size):
        d = C.shape[0]
        z=dist.MultivariateNormal(torch.zeros(d,dtype=torch.float32),C).expand([sample_size]).sample()
        x=dist.Chi2(df).expand([sample_size,1]).sample()
        return torch.abs(z/x)
    
    
    def _return_largest_eigenvector(self, cov_mat):
        eig_vals,eig_vecs = torch.symeig(cov_mat,eigenvectors=True)
        if (eig_vals==eig_vals.max()).sum()>1:
            top_vecs=eig_vecs[:,eig_vals==eig_vals.max()]
            return top_vecs[:,torch.randperm(top_vecs.shape[1])][:,0]
        else:            
            return torch.abs(eig_vecs[:,eig_vals==eig_vals.max()].flatten())
    
    
    def __init__(self,numPhenotypes,numLatentDimensions,numCatList=[],sparsityRate = 0.0,useMonotonic=True,**kwargs):
        allKeywordArgs = list(kwargs.keys()) 
        self.numPhenotypes=numPhenotypes
        self.numLatentDimensions=numLatentDimensions
        self.numCatList=numCatList
        
        if 'latentPhenotypeEffectsPrior' not in allKeywordArgs:
            self.latentPhenotypeEffectsPrior=torch.tensor([1.0,2.0],dtype=torch.float32)
        else:
            self.latentPhenotypeEffectsPrior=torch.tensor(kwargs['latentPhenotypeEffectsPrior'],dtype=torch.float32)
            
        if 'catCovEffectPriors' not in allKeywordArgs:
            self.catCovEffectPriors=torch.tensor([0.0,0.5],dtype=torch.float32)
        else:
            self.catCovEffectPriors=torch.tensor(kwargs['catCovEffectPriors'],dtype=torch.float32)
            
        if 'interceptPriors' not in allKeywordArgs:
            self.interceptPriors=torch.tensor([-3.0,1.0],dtype=torch.float32)
        else:
            self.interceptPriors=torch.tensor(kwargs['interceptPriors'],dtype=torch.float32)
            
        if 'outlierDegreeFreedom' not in allKeywordArgs:
            self.outlierDegreeFreedom=2.0*self.numLatentDimensions
        else:
            self.outlierDegreeFreedom=kwargs['outlierDegreeFreedom']
            
        if 'outlierCovarianceSparsity' not in allKeywordArgs:
            self.outlierCovarianceSparsity=(self.numLatentDimensions-1.0)/self.numLatentDimensions
        else:
            self.outlierCovarianceSparsity=kwargs['outlierCovarianceSparsity']

        if 'anchorDxNoisePrior' not in allKeywordArgs:
            self.anchorDxNoisePrior=torch.tensor([1.0,1.0],dtype=torch.float32)
        else:
            self.anchorDxNoisePrior=torch.tensor(kwargs['anchorDxNoisePrior'],dtype=torch.float32)
        
        if 'anchorDxThresholdPrior' not in allKeywordArgs:
            
            self.anchorDxThresholdPrior=torch.tensor(infer_liability_CI([0.00001,0.01]),dtype=torch.float32)
        else:
            self.anchorDxThresholdPrior=torch.tensor(infer_liability_CI(kwargs['anchorDxThresholdPrior']),dtype=torch.float32)
        
        #first construct the latent phenotype effects matrix
        if 'latentPhenotypeEffects' not in allKeywordArgs:
            
            self.latentPhenotypeEffectsPrecision=dist.Gamma(self.latentPhenotypeEffectsPrior[0],self.latentPhenotypeEffectsPrior[1]).expand([self.numLatentDimensions,1]).sample()
            if useMonotonic:
                self.latentPhenotypeEffects = dist.Exponential(torch.sqrt(self.latentPhenotypeEffectsPrecision)).expand([self.numLatentDimensions,self.numPhenotypes]).sample()
            else:
                self.latentPhenotypeEffects = dist.Normal(0.0,torch.sqrt(1.0/self.latentPhenotypeEffectsPrecision)).expand([self.numLatentDimensions,self.numPhenotypes]).sample()
                
            if sparsityRate>0.0:
                mask = dist.Bernoulli(sparsityRate).expand([self.numLatentDimensions,self.numPhenotypes]).sample()
                self.latentPhenotypeEffects[mask.to(dtype=torch.long)==1]=0.0
        else:
            self.latentPhenotypeEffects=torch.tensor(kwargs['latentPhenotypeEffects'])
            assert self.latentPhenotypeEffects.shape[0]==self.numLatentDimensions, "Provided latentPhenotypeEffects matrix does not match numLatentDimensions"
            assert self.latentPhenotypeEffects.shape[1]==self.numPhenotypes, "Provided latentPhenotypeEffects matrix does not match numPhenotypes"
            
            
        self.anchorDxNoise = dist.Gamma(self.anchorDxNoisePrior[0],self.anchorDxNoisePrior[1]).sample()
        self.anchorDxThreshold = dist.Normal(self.anchorDxThresholdPrior[0],self.anchorDxThresholdPrior[1]).sample()
        
        self.outlierCovarianceMatrix = self._gen_rand_cov_mat(self.numLatentDimensions,self.outlierCovarianceSparsity)
        self.anchorDxMap = self._return_largest_eigenvector(self.outlierCovarianceMatrix)
        
        self.intercepts = dist.Normal(self.interceptPriors[0],self.interceptPriors[1]).expand([self.numPhenotypes]).sample()
        
        if sum(self.numCatList)>0:
            self.covEffects = pyro.sample("covEffects",dist.Normal(self.priorParamDict['covEffects']['mean'],self.priorParamDict['covEffects']['scale']).expand([self.numObsTraits]).to_event(1))
            
    def GenerateClinicalData(self,numSamples,outlierFraction = 0.1):
        numIndependentSamples = int(np.floor(numSamples*(1.0-outlierFraction)))
        numOutlierSamples = numSamples-numIndependentSamples
        
        latentPhenotypes_Independent = dist.Normal(0.0,1.0).expand([numIndependentSamples,self.numLatentDimensions]).sample()
        latentPhenotypes_Outliers = self._folded_multivariate_TDist_RVs(self.outlierCovarianceMatrix,self.outlierDegreeFreedom,numOutlierSamples)
        latentPhenotypes = torch.cat((latentPhenotypes_Independent,latentPhenotypes_Outliers),dim=0)
        latentPhenotypes=latentPhenotypes[torch.randperm(latentPhenotypes.shape[0]),:]
        
        cat_cov_list = []
        if sum(self.numCatList)>0:
            for n_cat in self.numCatList:
                cat_cov_list+=[random_catcov(n_cat,numSamples)]
        mean_dx_rates = torch.mm(latentPhenotypes,self.latentPhenotypeEffects)+self.intercepts
        obs_data = dist.Bernoulli(dist.Normal(0.0,1.0).cdf(mean_dx_rates)).sample()
        output={}
        output['latent_phenotypes'] = latentPhenotypes
        output['incidence_data'] = obs_data
        output['covariate_data'] = cat_cov_list
        output['model_params'] = {}
        output['model_params']['intercepts']=self.intercepts
        output['model_params']['latentPhenotypeEffects']=self.latentPhenotypeEffects
        output['model_params']['latentPhenotypeEffectsPrecision']=self.latentPhenotypeEffectsPrecision
        
        if sum(self.numCatList)>0:
             output['model_params']['covEffects']=self.covEffects
             
        return output
        
    def GenerateAnchoringDx(self,latentPhenotypes):
        assert torch.is_tensor(latentPhenotypes),"Input must be an array of tensors"
        collapsedPheno = (latentPhenotypes*self.anchorDxMap).sum(dim=1,keepdim=True)
        dx_prob = dist.Normal(0.0,1.0).cdf((collapsedPheno+self.anchorDxThreshold)/self.anchorDxNoise)
        obs_data = dist.Bernoulli(dx_prob).sample()
        output={}
        output['anchor_dx_data']=obs_data
        output['model_params'] = {}
        output['model_params']['anchorDxMap']=self.anchorDxMap
        output['model_params']['anchorDxThreshold']=self.anchorDxThreshold
        output['model_params']['anchorDxNoise']=self.anchorDxNoise
        return output
    
        
if __name__=='__main__':
    
    simulator = ClinicalDataSimulator(20,2,sparsityRate = 0.5)
    simData=simulator.GenerateClinicalData(1000)
    simAnchors = simulator.GenerateAnchoringDx(simData['latent_phenotypes'])
#        
        
        
        