#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:21:44 2019

@author: davidblair
"""
import torch
import pyro
import pyro.distributions as dist
import numpy as np
from vlpi.utils.UtilityFunctions import random_catcov,infer_liability_CI


class ClinicalDataSimulator:


    def _generate_orthogonal_latent_pheno_params(self):
        effect_precisions=dist.Gamma(self.latentPhenotypeEffectsPrior[0],self.latentPhenotypeEffectsPrior[1]).expand([self.numLatentDimensions]).sample()

        component_assignments = torch.randint(low=0,high=self.numLatentDimensions,size=(self.numPhenotypes,))
        pheno_effects=torch.zeros((self.numLatentDimensions,self.numPhenotypes),dtype=torch.float32)

        for i in range(self.numLatentDimensions):
            if self.useMonotonic:
                pheno_effects[i,component_assignments==i]=dist.Exponential(torch.sqrt(effect_precisions[i])).expand([1,(component_assignments==i).sum()]).sample()
            else:
                pheno_effects[i,component_assignments==i]=dist.Normal(0.0,1.0/torch.sqrt(effect_precisions[i])).expand([1,(component_assignments==i).sum()]).sample()
        return effect_precisions,pheno_effects

    
    def _generate_dense_latent_pheno_params(self):
        effect_precisions=dist.Gamma(self.latentPhenotypeEffectsPrior[0],self.latentPhenotypeEffectsPrior[1]).expand([self.numLatentDimensions]).sample()
        pheno_effects=torch.zeros((self.numLatentDimensions,self.numPhenotypes),dtype=torch.float32)
        for i in range(self.numLatentDimensions):
            if self.useMonotonic:
                pheno_effects[i]=dist.Exponential(torch.sqrt(effect_precisions[i])).expand([1,self.numPhenotypes]).sample()
            else:
                pheno_effects[i]=dist.Normal(0.0,1.0/torch.sqrt(effect_precisions[i])).expand([1,self.numPhenotypes]).sample()
        
        return effect_precisions,pheno_effects
    
    
    def _generate_sparse_latent_pheno_params(self,sparsity_rate):
        effect_precisions,pheno_effects = self._generate_dense_latent_pheno_params()
        mask = dist.Bernoulli(probs = torch.ones(pheno_effects.shape)*(1.0-sparsity_rate)).sample().to(torch.float)
        return effect_precisions,pheno_effects*mask
        
        
        
    def __init__(self,numPhenotypes,numLatentDimensions,numCatList=[],useMonotonic=True,**kwargs):
        allKeywordArgs = list(kwargs.keys())
        self.numPhenotypes=numPhenotypes
        self.numLatentDimensions=numLatentDimensions
        self.numCatList=numCatList
        self.useMonotonic=useMonotonic

        if 'latentPhenotypeEffectsPrior' not in allKeywordArgs:
            self.latentPhenotypeEffectsPrior=torch.tensor([1.0,1.0],dtype=torch.float32)
        else:
            self.latentPhenotypeEffectsPrior=torch.tensor(kwargs['latentPhenotypeEffectsPrior'],dtype=torch.float32)

        if 'catCovEffectPriors' not in allKeywordArgs:
            self.catCovEffectPriors=torch.tensor([0.0,0.5],dtype=torch.float32)
        else:
            self.catCovEffectPriors=torch.tensor(kwargs['catCovEffectPriors'],dtype=torch.float32)


        if 'interceptPriors' not in allKeywordArgs:
            self.interceptPriors=torch.tensor([-2.5,1.0],dtype=torch.float32)
        else:
            self.interceptPriors=torch.tensor(kwargs['interceptPriors'],dtype=torch.float32)


        if 'numTargetDiseaseComponents' not in allKeywordArgs:
            self.numTargetDiseaseComponents=np.random.randint(1,self.numLatentDimensions+1)
        else:
            self.numTargetDiseaseComponents=kwargs['numTargetDiseaseComponents']



        if 'targetDxNoisePrior' not in allKeywordArgs:
            self.targetDxNoisePrior=torch.tensor([1.0,1.0],dtype=torch.float32)
        else:
            self.targetDxNoisePrior=torch.tensor(kwargs['targetDxNoisePrior'],dtype=torch.float32)

        if 'targetDxThresholdPrior' not in allKeywordArgs:

            self.targetDxThresholdPrior=torch.tensor(infer_liability_CI([0.00001,0.01]),dtype=torch.float32)
        else:
            self.targetDxThresholdPrior=torch.tensor(infer_liability_CI(kwargs['targetDxThresholdPrior']),dtype=torch.float32)

        #first construct the latent phenotype effects matrix
        if 'sparsityRate' not in allKeywordArgs:
            sparsityRate = 0.0
        else:
            sparsityRate=kwargs['sparsityRate']
            assert sparsityRate>0.0 and sparsityRate <1.0,"Sparsity rate must lie within (0.0,1.0)"
            
        if 'orthogonalLatentPhenotypes' not in allKeywordArgs:
            orthogonalLatentPhenotypes=False
        else:
            orthogonalLatentPhenotypes=kwargs['orthogonalLatentPhenotypes']
            
        
        
        
        if 'latentPhenotypeEffects' not in allKeywordArgs:
            if sparsityRate > 0.0:
                assert orthogonalLatentPhenotypes==False,"Cannot simultaneously use sparse and orthogonal latent phenotypes"
                self.latentPhenotypeEffectsPrecision, self.latentPhenotypeEffects=self._generate_sparse_latent_pheno_params(sparsityRate)
            elif orthogonalLatentPhenotypes==True:
                self.latentPhenotypeEffectsPrecision, self.latentPhenotypeEffects=self._generate_orthogonal_latent_pheno_params()
                
            else:
                    self.latentPhenotypeEffectsPrecision, self.latentPhenotypeEffects=self._generate_dense_latent_pheno_params()
        else:
            self.latentPhenotypeEffects=torch.tensor(kwargs['latentPhenotypeEffects'])
            assert self.latentPhenotypeEffects.shape[0]==self.numLatentDimensions, "Provided latentPhenotypeEffects matrix does not match numLatentDimensions"
            assert self.latentPhenotypeEffects.shape[1]==self.numPhenotypes, "Provided latentPhenotypeEffects matrix does not match numPhenotypes"




        self.targetDxNoise = dist.Gamma(self.targetDxNoisePrior[0],self.targetDxNoisePrior[1]).sample()
        self.targetDxThreshold = dist.Normal(self.targetDxThresholdPrior[0],self.targetDxThresholdPrior[1]).sample()

        associated_components = np.random.choice(np.arange(self.numLatentDimensions),size=self.numTargetDiseaseComponents,replace=False)
        self.targetDxMap = torch.zeros((self.numLatentDimensions),dtype=torch.float32)
        self.targetDxMap[associated_components]=torch.sqrt(dist.Dirichlet(torch.ones(self.numTargetDiseaseComponents)).sample())


        self.intercepts = dist.Normal(self.interceptPriors[0],self.interceptPriors[1]).expand([self.numPhenotypes]).sample()

        if sum(self.numCatList)>0:
            self.covEffects = pyro.sample("covEffects",dist.Normal(self.catCovEffectPriors[0],self.catCovEffectPriors[1]).expand([sum(numCatList),self.numPhenotypes]))

    def GenerateClinicalData(self,numSamples):

        latentPhenotypes= dist.Normal(0.0,1.0).expand([numSamples,self.numLatentDimensions]).sample()

        cat_cov_list = []
        if sum(self.numCatList)>0:
            for n_cat in self.numCatList:
                cat_cov_list+=[random_catcov(n_cat,numSamples,device='cpu')]
        mean_dx_rates = torch.mm(latentPhenotypes,self.latentPhenotypeEffects)+self.intercepts
        obs_data = dist.Bernoulli(dist.Normal(0.0,1.0).cdf(mean_dx_rates)).sample()
        output={}
        output['latent_phenotypes'] = latentPhenotypes
        output['incidence_data'] = obs_data
        output['covariate_data'] = cat_cov_list
        output['model_params'] = {}
        output['model_params']['intercepts']=self.intercepts
        output['model_params']['latentPhenotypeEffects']=self.latentPhenotypeEffects

        if sum(self.numCatList)>0:
             output['model_params']['covEffects']=self.covEffects

        return output

    def GenerateTargetDx(self,latentPhenotypes):
        assert torch.is_tensor(latentPhenotypes),"Input must be an array of tensors"
        collapsedPheno = (latentPhenotypes*self.targetDxMap).sum(dim=1,keepdim=True)
        dx_prob = dist.Normal(0.0,1.0).cdf((collapsedPheno+self.targetDxThreshold)/self.targetDxNoise)
        obs_data = dist.Bernoulli(dx_prob).sample()
        output={}
        output['target_dx_data']=obs_data
        output['model_params'] = {}
        output['model_params']['targetDxMap']=self.targetDxMap
        output['model_params']['targetDxThreshold']=self.targetDxThreshold
        output['model_params']['targetDxNoise']=self.targetDxNoise
        return output


if __name__=='__main__':

    simulator = ClinicalDataSimulator(20,2,sparsityRate=0.5)
    simData=simulator.GenerateClinicalData(10000)
    simLabels = simulator.GenerateTargetDx(simData['latent_phenotypes'])


