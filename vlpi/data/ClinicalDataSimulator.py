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


    def _sample_outliers(self,sample_size):

        mean_vals =  dist.Normal(0.0,1.0).icdf(dist.Uniform(self.outlierPercentile,1.0).expand([sample_size,1]).sample())*self.labelDxMap
        scale_vec=torch.ones((1,self.numLatentDimensions),dtype=torch.float32)
        scale_vec[0,self.labelDxMap!=0.0]=self.outlierNoise
        return dist.Normal(mean_vals,scale_vec).sample()



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


        if 'outlierPercentile' not in allKeywordArgs:
            self.outlierPercentile=0.999
        else:
            self.outlierPercentile=kwargs['outlierPercentile']

        if 'outlierNoise' not in allKeywordArgs:
            self.outlierNoise = 0.1
        else:
            self.outlierNoise=kwargs['outlierNoise']

        if 'numMendelianComponents' not in allKeywordArgs:
            self.numMendelianComponents=np.random.randint(1,self.numLatentDimensions+1)
        else:
            self.numMendelianComponents=kwargs['numMendelianComponents']



        if 'labelDxNoisePrior' not in allKeywordArgs:
            self.labelDxNoisePrior=torch.tensor([1.0,1.0],dtype=torch.float32)
        else:
            self.labelDxNoisePrior=torch.tensor(kwargs['labelDxNoisePrior'],dtype=torch.float32)

        if 'labelDxThresholdPrior' not in allKeywordArgs:

            self.labelDxThresholdPrior=torch.tensor(infer_liability_CI([0.00001,0.01]),dtype=torch.float32)
        else:
            self.labelDxThresholdPrior=torch.tensor(infer_liability_CI(kwargs['labelDxThresholdPrior']),dtype=torch.float32)

        #first construct the latent phenotype effects matrix
        if 'latentPhenotypeEffects' not in allKeywordArgs:
            self.latentPhenotypeEffectsPrecision, self.latentPhenotypeEffects=self._generate_orthogonal_latent_pheno_params()
        else:
            self.latentPhenotypeEffects=torch.tensor(kwargs['latentPhenotypeEffects'])
            assert self.latentPhenotypeEffects.shape[0]==self.numLatentDimensions, "Provided latentPhenotypeEffects matrix does not match numLatentDimensions"
            assert self.latentPhenotypeEffects.shape[1]==self.numPhenotypes, "Provided latentPhenotypeEffects matrix does not match numPhenotypes"




        self.labelDxNoise = dist.Gamma(self.labelDxNoisePrior[0],self.labelDxNoisePrior[1]).sample()
        self.labelDxThreshold = dist.Normal(self.labelDxThresholdPrior[0],self.labelDxThresholdPrior[1]).sample()

        associated_components = np.random.choice(np.arange(self.numLatentDimensions),size=self.numMendelianComponents,replace=False)
        self.labelDxMap = torch.zeros((self.numLatentDimensions),dtype=torch.float32)
        self.labelDxMap[associated_components]=torch.sqrt(dist.Dirichlet(torch.ones(self.numMendelianComponents)).sample())


        self.intercepts = dist.Normal(self.interceptPriors[0],self.interceptPriors[1]).expand([self.numPhenotypes]).sample()

        if sum(self.numCatList)>0:
            self.covEffects = pyro.sample("covEffects",dist.Normal(self.catCovEffectPriors[0],self.catCovEffectPriors[1]).expand([sum(numCatList),self.numPhenotypes]))

    def GenerateClinicalData(self,numSamples,outlierFraction):
        numIndependentSamples = int(np.floor(numSamples*(1.0-outlierFraction)))
        numOutlierSamples = numSamples-numIndependentSamples

        latentPhenotypes_Independent = dist.Normal(0.0,1.0).expand([numIndependentSamples,self.numLatentDimensions]).sample()
        latentPhenotypes_Outliers = self._sample_outliers(numOutlierSamples)

        outlier_inds = torch.zeros((numIndependentSamples,1),dtype=torch.long)
        outlier_inds=torch.cat((outlier_inds,torch.ones((numOutlierSamples,1),dtype=torch.long)))

        new_order=torch.randperm(outlier_inds.shape[0])

        latentPhenotypes = torch.cat((latentPhenotypes_Independent,latentPhenotypes_Outliers),dim=0)
        latentPhenotypes=latentPhenotypes[new_order,:]
        outlier_inds=outlier_inds[new_order,:]

        cat_cov_list = []
        if sum(self.numCatList)>0:
            for n_cat in self.numCatList:
                cat_cov_list+=[random_catcov(n_cat,numSamples,device='cpu')]
        mean_dx_rates = torch.mm(latentPhenotypes,self.latentPhenotypeEffects)+self.intercepts
        obs_data = dist.Bernoulli(dist.Normal(0.0,1.0).cdf(mean_dx_rates)).sample()
        output={}
        output['latent_phenotypes'] = latentPhenotypes
        output['is_outlier'] = outlier_inds
        output['incidence_data'] = obs_data
        output['covariate_data'] = cat_cov_list
        output['model_params'] = {}
        output['model_params']['intercepts']=self.intercepts
        output['model_params']['latentPhenotypeEffects']=self.latentPhenotypeEffects
        output['model_params']['latentPhenotypeEffectsPrecision']=self.latentPhenotypeEffectsPrecision

        if sum(self.numCatList)>0:
             output['model_params']['covEffects']=self.covEffects

        return output

    def GenerateLabelDx(self,latentPhenotypes):
        assert torch.is_tensor(latentPhenotypes),"Input must be an array of tensors"
        collapsedPheno = (latentPhenotypes*torch.sqrt(self.labelDxMap)).sum(dim=1,keepdim=True)
        dx_prob = dist.Normal(0.0,1.0).cdf((collapsedPheno+self.labelDxThreshold)/self.labelDxNoise)
        obs_data = dist.Bernoulli(dx_prob).sample()
        output={}
        output['label_dx_data']=obs_data
        output['model_params'] = {}
        output['model_params']['labelDxMap']=self.labelDxMap
        output['model_params']['labelDxThreshold']=self.labelDxThreshold
        output['model_params']['labelDxNoise']=self.labelDxNoise
        return output


if __name__=='__main__':

    simulator = ClinicalDataSimulator(20,2)
    simData=simulator.GenerateClinicalData(10000,0.01)
    simLabels = simulator.GenerateLabelDx(simData['latent_phenotypes'])

    centroid=simData['latent_phenotypes'].numpy().mean(axis=0)
    true_euclid_dist = np.sqrt(np.sum((simData['latent_phenotypes'].numpy()-centroid)**2,axis=1,keepdims=True))

    pred_disease_liability = np.sum(simLabels['model_params']['labelDxMap'].numpy()*simData['latent_phenotypes'].numpy(),axis=1,keepdims=True)

#    plt.plot(true_euclid_dist[simData['is_outlier'].numpy()==1],pred_disease_liability[simData['is_outlier'].numpy()==1],'o')
#    plt.plot(true_euclid_dist[simData['is_outlier'].numpy()==0],pred_disease_liability[simData['is_outlier'].numpy()==0],'o')
##
#
    plt.plot(simData['latent_phenotypes'][simData['is_outlier'].flatten()==0,0].numpy(),simData['latent_phenotypes'][simData['is_outlier'].flatten()==0,1].numpy(),'o')
    plt.plot(simData['latent_phenotypes'][simData['is_outlier'].flatten()==1,0].numpy(),simData['latent_phenotypes'][simData['is_outlier'].flatten()==1,1].numpy(),'o')
