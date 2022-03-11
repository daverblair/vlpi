#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:21:44 2019

@author: davidblair
"""
import torch
import pyro
import pyro.distributions as dist
from vlpi.utils.UtilityFunctions import random_catcov


class ClinicalDataSimulator:


    def _generate_orthogonal_latent_pheno_params(self):
        effect_precisions=dist.Gamma(self.latentPhenotypeEffectsPrior[0],self.latentPhenotypeEffectsPrior[1]).expand([self.numLatentDimensions]).sample()

        component_assignments = torch.randint(low=0,high=self.numLatentDimensions,size=(self.numObsPhenotypes,))
        pheno_effects=torch.zeros((self.numLatentDimensions,self.numObsPhenotypes),dtype=torch.float32)

        for i in range(self.numLatentDimensions):
            if self.useMonotonic:
                pheno_effects[i,component_assignments==i]=dist.Exponential(torch.sqrt(effect_precisions[i])).expand([1,(component_assignments==i).sum()]).sample()
            else:
                pheno_effects[i,component_assignments==i]=dist.Normal(0.0,1.0/torch.sqrt(effect_precisions[i])).expand([1,(component_assignments==i).sum()]).sample()
        return effect_precisions,pheno_effects


    def _generate_dense_latent_pheno_params(self):
        effect_precisions=dist.Gamma(self.latentPhenotypeEffectsPrior[0],self.latentPhenotypeEffectsPrior[1]).expand([self.numLatentDimensions]).sample()
        pheno_effects=torch.zeros((self.numLatentDimensions,self.numObsPhenotypes),dtype=torch.float32)
        for i in range(self.numLatentDimensions):
            if self.useMonotonic:
                pheno_effects[i]=dist.Exponential(torch.sqrt(effect_precisions[i])).expand([1,self.numObsPhenotypes]).sample()
            else:
                pheno_effects[i]=dist.Normal(0.0,1.0/torch.sqrt(effect_precisions[i])).expand([1,self.numObsPhenotypes]).sample()

        return effect_precisions,pheno_effects


    def _generate_sparse_latent_pheno_params(self,sparsity_rate):
        effect_precisions,pheno_effects = self._generate_dense_latent_pheno_params()
        mask = dist.Bernoulli(probs = torch.ones(pheno_effects.shape)*(1.0-sparsity_rate)).sample().to(torch.float)
        return effect_precisions,pheno_effects*mask



    def __init__(self,numObsPhenotypes,numLatentDimensions,targetDisPrev,isOutlier=False,includeModifierEffects=True,numCatList=[],useMonotonic=True,**kwargs):
        """
        Simulator Class for the variational latent phenotype model.

        Simulated data is based on the simple model outlined in Blair et al.

        Parameters
        ----------
        numObsPhenotypes : int
            Number of observed symptoms/phenotypes to simulate.
        numLatentDimensions : int
            Number of latent/cryptic phenotypes to include into the model.
        targetDisPrev : float
            Prevalence of target/Mendelian disease.
        isOutlier : bool, optional
            Determines whether the target disease is a phenotypic outlier. The default is False.
        includeModifierEffects : bool, optional
            Determines whether or not to include severity-based modifier effects. The default is True.
        numCatList : [int], optional
            List of intergers providing number of categories for each covariate. Total number of covariates: len(numCatList). The default is [].
        useMonotonic : boold, optional
            Whether to enforce monotonicity between latent phenotypes and risk for observed symptoms. The default is True.
        **kwargs : Multiple
            Extensive parameter modification choices. See source code for details. Baseline choices match those used in simulation analyses.

        Returns
        -------
        None.

        """
        allKeywordArgs = list(kwargs.keys())
        self.numObsPhenotypes=numObsPhenotypes
        self.numLatentDimensions=numLatentDimensions
        self.numCatList=numCatList
        self.useMonotonic=useMonotonic
        self.targetDisPrev=torch.tensor(targetDisPrev)
        self.targetDisComponent=torch.randperm(self.numLatentDimensions)[0]
        self.isOutlier=isOutlier
        self.includeModifierEffects=includeModifierEffects

        if 'latentPhenotypeEffectsPrior' not in allKeywordArgs:
            self.latentPhenotypeEffectsPrior=torch.tensor([1.0,1.0],dtype=torch.float32)
        else:
            self.latentPhenotypeEffectsPrior=torch.tensor(kwargs['latentPhenotypeEffectsPrior'],dtype=torch.float32)

        if 'catCovEffectPriors' not in allKeywordArgs:
            self.catCovEffectPriors=torch.tensor([0.0,0.5],dtype=torch.float32)
        else:
            self.catCovEffectPriors=torch.tensor(kwargs['catCovEffectPriors'],dtype=torch.float32)


        if 'interceptPriors' not in allKeywordArgs:
            self.interceptPriors=torch.tensor([-3.0,2.0],dtype=torch.float32)
        else:
            self.interceptPriors=torch.tensor(kwargs['interceptPriors'],dtype=torch.float32)


        if 'targetDisEffectParam' not in allKeywordArgs:
            self.targetDisEffectParam=dist.Normal(0.0,1.0).icdf(1.0-self.targetDisPrev)
        else:
            self.targetDisEffectParam=torch.tensor(kwargs['targetDisEffectParam'],dtype=torch.float32)


        if 'latentPhenotypePriors' not in allKeywordArgs:
            self.latentPhenotypePrior=torch.tensor([0.0,1.0],dtype=torch.float32)
            self.latentPhenotypeOutlierPrior=torch.tensor([0.0,0.0001],dtype=torch.float32)
        else:
            self.latentPhenotypePrior=torch.tensor(kwargs['latentPhenotypePriors'][0],dtype=torch.float32)
            self.latentPhenotypeOutlierPrior=torch.tensor(kwargs['latentPhenotypePriors'][1],dtype=torch.float32)
            print("Warning: By specifying custom latent phenotype priors, you can break the relationship between outliers and spectrums.")

        if self.includeModifierEffects:
            if 'modifierEffectPrior' not in allKeywordArgs:
                self.modifierEffectPrior=torch.tensor([0.0,1.0],dtype=torch.float32)
            else:
                self.modifierEffectPrior=torch.tensor(kwargs['modifierEffectPrior'],dtype=torch.float32)

            if 'modifierEffectThreshold' not in allKeywordArgs:
                self.modifierEffectThreshold=self.targetDisEffectParam
            else:
                self.modifierEffectThreshold=torch.tensor(kwargs['modifierEffectThreshold'],dtype=torch.float32)

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
            assert self.latentPhenotypeEffects.shape[1]==self.numObsPhenotypes, "Provided latentPhenotypeEffects matrix does not match numObsPhenotypes"


        #now the intercepts for the observed symptoms
        self.intercepts = dist.Normal(self.interceptPriors[0],self.interceptPriors[1]).expand([self.numObsPhenotypes]).sample()

        #covariate effects (optional)
        if sum(self.numCatList)>0:
            self.covEffects = pyro.sample("covEffects",dist.Normal(self.catCovEffectPriors[0],self.catCovEffectPriors[1]).expand([sum(numCatList),self.numObsPhenotypes]))

    def GenerateClinicalData(self,numSamples):
        """

        Simulates clinical data and returns a dictionary of the results.

        Parameters
        ----------
        numSamples : int
            Number of samples to simulate.

        Returns
        -------
        output : dict
            Dictionary containing data from simulation.

            output['latent_phenotypes']: Siumulated Latent Phenotypes
            output['incidence_data']: Binary symptom matrix
            output['covariate_data']: List of arrays containing categorical covariates.
            output['target_dis_dx']: Mendelian/target disease diagnoses
            output['model_params']: Dictionary of model parameters
            output['model_params']['intercepts']: Vector of intercepts for symptom risk function.
            output['model_params']['latentPhenotypeEffects']: Matrix of symptom risk parameters (loading matrix)

        """

        latentPhenotypes=torch.zeros([numSamples,self.numLatentDimensions],dtype=torch.float32)

        for i in range(self.numLatentDimensions):
            if i!=self.targetDisComponent:
                latentPhenotypes[:,i]=dist.Normal(self.latentPhenotypePrior[0],self.latentPhenotypePrior[1]).expand([numSamples]).sample()
            else:
                targetDisSubjects=dist.Bernoulli(self.targetDisPrev).expand([numSamples]).sample()
                if self.isOutlier==False:
                    latentPhenotypes[:,i]=dist.Normal(self.latentPhenotypePrior[0],self.latentPhenotypePrior[1]).expand([numSamples]).sample()+targetDisSubjects*self.targetDisEffectParam
                else:
                    latentPhenotypes[:,i]=dist.Normal(self.latentPhenotypeOutlierPrior[0],self.latentPhenotypeOutlierPrior[1]).expand([numSamples]).sample()+targetDisSubjects*self.targetDisEffectParam
                if self.includeModifierEffects:
                    latentPhenotypes[:,i]=latentPhenotypes[:,i]+dist.Normal(self.modifierEffectThreshold,1.0).cdf(latentPhenotypes[:,i])*dist.Normal(self.modifierEffectPrior[0],self.modifierEffectPrior[1]).expand([numSamples]).sample()


        cat_cov_list = []
        if sum(self.numCatList)>0:
            for n_cat in self.numCatList:
                cat_cov_list+=[random_catcov(n_cat,numSamples,device='cpu')]
        mean_dx_rates = torch.mm(latentPhenotypes,self.latentPhenotypeEffects)+self.intercepts
        obs_data = dist.Bernoulli(torch.sigmoid(mean_dx_rates)).sample()
        output={}
        output['latent_phenotypes'] = latentPhenotypes
        output['incidence_data'] = obs_data
        output['covariate_data'] = cat_cov_list
        output['target_dis_dx'] = targetDisSubjects
        output['model_params'] = {}
        output['model_params']['intercepts']=self.intercepts
        output['model_params']['latentPhenotypeEffects']=self.latentPhenotypeEffects

        if sum(self.numCatList)>0:
             output['model_params']['covEffects']=self.covEffects

        return output


if __name__=='__main__':

    simulator = ClinicalDataSimulator(20,2,0.001,isOutlier=False)
    simData=simulator.GenerateClinicalData(100000)
    # simLabels = simulator.GenerateTargetDx(simData['latent_phenotypes'])
