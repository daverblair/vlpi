#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:31:38 2019

@author: davidblair
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO
from typing import Iterable
from vlpi.Encoders import MeanScaleEncoder
from vlpi.utils import random_catcov,build_onehot_arrays

import copy

class _vlpiModel(nn.Module):
    def _computeCovEffects(self,cat_cov_list,cov_effect_array):
        one_hot_cat_list = build_onehot_arrays(cat_cov_list,self.numCatList,self.dropLinearCovariateColumn)
        return torch.mm(torch.cat((*one_hot_cat_list,),dim=-1),torch.transpose(cov_effect_array,0,1))


    def SwitchDevice(self,new_compute_device):
        self.compute_device=new_compute_device

        for varName,pDict in self.priorParamDict.items():
            for vType,t in pDict.items():
                self.priorParamDict[varName][vType]=t.to(self.compute_device)

        for varName,pDict in self.posteriorParamDict.items():
            for vType,t in pDict.items():
                self.posteriorParamDict[varName][vType]=t.to(self.compute_device)
        self.to(self.compute_device)


    def _copyModelPosteriorParams(self):
        new_dict={}
        for variable,p_dict in self.posteriorParamDict.items():
            new_dict[variable]={}
            for param_type,p_vals in p_dict.items():
                new_dict[variable][param_type]=p_vals.detach()
        return new_dict


    def _copyModelPriorParams(self):
        new_dict={}
        for variable,p_dict in self.priorParamDict.items():
            new_dict[variable]={}
            for param_type,p_vals in p_dict.items():
                new_dict[variable][param_type]=p_vals.detach()
        return new_dict

    def _encoder_only_model(self,obs_data=None, cat_cov_list=None,label_dx=None, sample_scores=None,numSamples=None,minibatch_scale=1.0):
        
        
        assert sample_scores is not None, "Sample scores must be included in order to train the encoder individually."
        assert obs_data is not None, "obs_data must be included when using the _encoder_only_model."
        pyro.module("encoder", self.encoder)

        numSamples=obs_data.shape[0]

        latent_vals = torch.zeros(sample_scores.shape,dtype=torch.float32,device=sample_scores.device)
        norm_vals = torch.arange(sample_scores.shape[0],device=sample_scores.device).to(dtype=torch.float32)
        norm_vals = dist.Normal(0.0,1.0).icdf(1.0-(norm_vals+1.0)/(norm_vals.shape[0]+1.0))
        for i in range(self.nLatentDim):
            rankings=sample_scores[:,i].argsort(dim=0,descending=True)
            latent_vals[rankings,i]=norm_vals

        #add noise to latent states to prevent overtraining the variance
        latent_vals+=dist.Normal(0.0,1.0).sample(latent_vals.shape)

        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("rank_pheno_plate",size=numSamples):
                z_mean,z_std = self.encoder(obs_data,*cat_cov_list)
                pyro.sample("rankPhenotypes",dist.Normal(z_mean, z_std).to_event(1),obs=latent_vals)


    def _encoder_only_guide(self,obs_data=None, cat_cov_list=None,label_dx=None, sample_scores=None,numSamples=None,minibatch_scale=1.0):
        assert sample_scores is not None, "Sample scores must be included in order to train the encoder individually."
        assert obs_data is not None, "obs_data must be included when using the _encoder_only_model."
        

    def __init__(self,numObsTraits:int, numCatList:Iterable[int],nLatentDim:int,latentPhenotypePriors={'latentPhenotypeEffectsPrecision':[1.0,1.0]},fixedEffectPriors={'intercepts':[0.0,5.0],'covariates_scale':3.0},**kwargs):




        super(_vlpiModel,self).__init__()
        self.numObsTraits=numObsTraits
        self.nLatentDim = nLatentDim
        self.numCatList=numCatList
        
        

        

        allKeywordArgs = list(kwargs.keys())
        
        if 'mappingFunction' not in allKeywordArgs:
            self.mappingFunction='Linear_Monotonic'
        else:
            assert kwargs['mappingFunction'] in ['Linear','Linear_Monotonic'], "Currently supported mapping model types for vLPM: Linear, Linear_Monotonic"
            self.mappingFunction=kwargs['mappingFunction']

        if 'computeDevice' not in allKeywordArgs:
            """
            Specifies compute device for model fitting, can also be specified later by
            calling SwitchDevice

            """
            self.compute_device=None
        else:
            self.compute_device=kwargs['computeDevice']

        if 'dropLinearCovariateColumn' not in allKeywordArgs:
            """
            Specifies model to drop one category from each covariate. Defaults to True.
            """
            self.dropLinearCovariateColumn=True
        else:
            self.dropLinearCovariateColumn=kwargs['dropLinearCovariateColumn']
            assert isinstance(self.dropLinearCovariateColumn,bool),"dropLinearCovariateColumn expects boolean value"
            
        

        if 'neuralNetworkHyperparameters' not in allKeywordArgs:
            self.encoderHyperparameters={'n_layers' : 2, 'n_hidden' : 128, 'dropout_rate': 0.2, 'use_batch_norm':True}

        else:
            self.encoderHyperparameters = kwargs['neuralNetworkHyperparameters']
            assert isinstance(self.encoderHyperparameters,dict),"Expects dictionary of encoder hyperparameters"
            assert set(self.encoderHyperparameters.keys())==set(['n_layers','n_hidden','dropout_rate','use_batch_norm']),"Encoder hyperparameters must include: 'n_layers','n_hidden','dropout_rate','use_batch_norm'"

        if self.dropLinearCovariateColumn:
            self.numCovParam = sum(self.numCatList)-len(self.numCatList)
        else:
            self.numCovParam = sum(self.numCatList)

        self.encoder=MeanScaleEncoder(self.numObsTraits, self.nLatentDim, n_cat_list= self.numCatList, **self.encoderHyperparameters)

        self.posteriorParamDict = {}
        self.priorParamDict={}

            
            
        self.priorParamDict['intercepts']={}
        self.priorParamDict['intercepts']['mean'] = torch.tensor(fixedEffectPriors['intercepts'][0],dtype=torch.float32)
        self.priorParamDict['intercepts']['scale'] = torch.tensor(fixedEffectPriors['intercepts'][1],dtype=torch.float32)
        self.posteriorParamDict['intercepts'] = {'mean':torch.ones(self.numObsTraits,dtype=torch.float32)*fixedEffectPriors['intercepts'][0],'scale':torch.ones(self.numObsTraits,dtype=torch.float32)*fixedEffectPriors['intercepts'][1]}


        if self.numCovParam>0:
            self.posteriorParamDict['covEffects']={'mean':torch.zeros(self.numObsTraits,self.numCovParam,dtype=torch.float32),'scale':torch.ones(self.numObsTraits,self.numCovParam,dtype=torch.float32)*fixedEffectPriors['covariates_scale']}
            self.priorParamDict['covEffects']={}
            self.priorParamDict['covEffects']['mean'] = torch.tensor(0.0,dtype=torch.float32)
            self.priorParamDict['covEffects']['scale'] = torch.tensor(fixedEffectPriors['covariates_scale'],dtype=torch.float32)



        self.priorParamDict['latentPhenotypeEffectsPrecision']={}
        self.priorParamDict['latentPhenotypeEffectsPrecision']['conc']=torch.tensor(latentPhenotypePriors['latentPhenotypeEffectsPrecision'][0],dtype=torch.float32)
        self.priorParamDict['latentPhenotypeEffectsPrecision']['rate']=torch.tensor(latentPhenotypePriors['latentPhenotypeEffectsPrecision'][1],dtype=torch.float32)
        self.posteriorParamDict['latentPhenotypeEffectsPrecision']= {'conc':torch.ones((self.nLatentDim,1),dtype=torch.float32)*latentPhenotypePriors['latentPhenotypeEffectsPrecision'][0],'rate':torch.ones((self.nLatentDim,1),dtype=torch.float32)*latentPhenotypePriors['latentPhenotypeEffectsPrecision'][1]}
        
        if self.mappingFunction=='Linear_Monotonic':
            self.posteriorParamDict['latentPhenotypeEffects'] = {'conc':torch.ones(self.nLatentDim,self.numObsTraits,dtype=torch.float32),'rate':torch.ones(self.nLatentDim,self.numObsTraits,dtype=torch.float32)}
        else:
            self.posteriorParamDict['latentPhenotypeEffects'] = {'mean':torch.zeros(self.nLatentDim,self.numObsTraits,dtype=torch.float32),'scale':torch.ones(self.nLatentDim,self.numObsTraits,dtype=torch.float32)}

 

        if self.compute_device is not None:
            self.SwitchDevice(self.compute_device)

        self.eval()
    


    def model(self, obs_data=None, cat_cov_list=None,label_dx=None,sample_scores=None,numSamples=None,minibatch_scale=1.0):
        if obs_data is not None:
            numSamples=obs_data.shape[0]
        else:
                #assert numSamples is not None, "Must provide either observed data or number of samples to simulate."
            if numSamples is None:
                numSamples=1000
                print('Warning: no arguments were given to Continuous.model. This should only be done during debugging.')

            cat_cov_list = []
            for n_cat in self.numCatList:
                cat_cov_list+=[random_catcov(n_cat,numSamples,device=self.compute_device)]


        with pyro.plate("intercept_plate",size=self.numObsTraits,dim=-1):
            intercepts = pyro.sample("intercepts",dist.Normal(self.priorParamDict['intercepts']['mean'],self.priorParamDict['intercepts']['scale']))
            
        if self.numCovParam>0:
            
            with pyro.plate("cov_plate",size=self.numObsTraits,dim=-1):
                covEffects = pyro.sample("covEffects",dist.Normal(self.priorParamDict['covEffects']['mean'],self.priorParamDict['covEffects']['scale']).expand([1,self.numCovParam]).to_event(1))

        with pyro.plate("prec_plate",size=self.nLatentDim,dim=-1):
            latentPhenotypePrecision = pyro.sample("latentPhenotypeEffectsPrecision",dist.Gamma(self.priorParamDict['latentPhenotypeEffectsPrecision']['conc'],self.priorParamDict['latentPhenotypeEffectsPrecision']['rate']).expand([self.nLatentDim,1]).to_event(1))

        if self.mappingFunction=='Linear_Monotonic':
            with pyro.plate("latent_pheno_effect_plate",size=self.nLatentDim,dim=-1):
                latentPhenotypeEffects = pyro.sample("latentPhenotypeEffects",dist.Exponential(torch.sqrt(latentPhenotypePrecision)).expand([self.nLatentDim,self.numObsTraits]).to_event(1))
        else:
            with pyro.plate("latent_pheno_effect_plate",size=self.nLatentDim,dim=-1):
                latentPhenotypeEffects = pyro.sample("latentPhenotypeEffects",dist.Normal(0.0,torch.sqrt(1.0/latentPhenotypePrecision)).expand([self.nLatentDim,self.numObsTraits]).to_event(1))

        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                latentPhenotypes=pyro.sample("latentPhenotypes",dist.Normal(torch.zeros(1,self.nLatentDim,dtype=torch.float32,device=self.compute_device),torch.ones(1,self.nLatentDim,dtype=torch.float32,device=self.compute_device)).to_event(1))
                liabilities = torch.mm(latentPhenotypes,latentPhenotypeEffects)+intercepts
                if self.numCovParam>0:
                    liabilities+=self._computeCovEffects(cat_cov_list,covEffects)
                mean_dx_rates=dist.Normal(0.0,1.0).cdf(liabilities)
            
                pyro.sample("obsTraitIncidence",dist.Bernoulli(mean_dx_rates).to_event(1),obs=obs_data)




    def guide(self,obs_data=None, cat_cov_list=None,label_dx=None,sample_scores=None,numSamples = None,minibatch_scale=1.0):
        if obs_data is not None:
            numSamples=obs_data.shape[0]

        else:
            pass

        if sample_scores is None:
            pyro.module("encoder", self.encoder)


        self.posteriorParamDict['intercepts']['mean']=pyro.param("interceptPosteriorMean",init_tensor=self.posteriorParamDict['intercepts']['mean'])
        self.posteriorParamDict['intercepts']['scale']=pyro.param("interceptPosteriorScale",init_tensor=self.posteriorParamDict['intercepts']['scale'],constraint=torch.distributions.constraints.positive)
        with pyro.plate("intercept_plate",size=self.numObsTraits,dim=-1):
            pyro.sample("intercepts",dist.Normal(self.posteriorParamDict['intercepts']['mean'] ,self.posteriorParamDict['intercepts']['scale']))
            
        if self.numCovParam>0:
            self.posteriorParamDict['covEffects']['mean']=pyro.param("covEffectsPosteriorMean",init_tensor=self.posteriorParamDict['covEffects']['mean'])
            self.posteriorParamDict['covEffects']['scale']=pyro.param("covEffectsPosteriorScale",init_tensor=self.posteriorParamDict['covEffects']['scale'],constraint=torch.distributions.constraints.positive)
            with pyro.plate("cov_plate",size=self.numObsTraits,dim=-1):
                pyro.sample("covEffects",dist.Normal(self.posteriorParamDict['covEffects']['mean'],self.posteriorParamDict['covEffects']['scale']).to_event(1))
            


        self.posteriorParamDict['latentPhenotypeEffectsPrecision']['conc']=pyro.param('latentPhenotypeEffectsPrecisionPosteriorConc',init_tensor=self.posteriorParamDict['latentPhenotypeEffectsPrecision']['conc'],constraint=torch.distributions.constraints.positive)
        self.posteriorParamDict['latentPhenotypeEffectsPrecision']['rate']=pyro.param('latentPhenotypeEffectsPrecisionPosteriorRate',init_tensor=self.posteriorParamDict['latentPhenotypeEffectsPrecision']['rate'],constraint=torch.distributions.constraints.positive)
        
        with pyro.plate("prec_plate",size=self.nLatentDim,dim=-1):
            pyro.sample("latentPhenotypeEffectsPrecision",dist.Gamma(self.posteriorParamDict['latentPhenotypeEffectsPrecision']['conc'],self.posteriorParamDict['latentPhenotypeEffectsPrecision']['rate']).expand([self.nLatentDim,1]).to_event(1))
        

        if self.mappingFunction=='Linear_Monotonic':
            self.posteriorParamDict['latentPhenotypeEffects']['conc']=pyro.param('latentPhenotypeEffectsPosteriorConc',init_tensor=self.posteriorParamDict['latentPhenotypeEffects']['conc'],constraint=torch.distributions.constraints.positive)
            self.posteriorParamDict['latentPhenotypeEffects']['rate']=pyro.param('latentPhenotypeEffectsPosteriorRates',init_tensor=self.posteriorParamDict['latentPhenotypeEffects']['rate'],constraint=torch.distributions.constraints.positive)
            
            with pyro.plate("latent_pheno_effect_plate",size=self.nLatentDim,dim=-1):
                pyro.sample("latentPhenotypeEffects",dist.Gamma(self.posteriorParamDict['latentPhenotypeEffects']['conc'],self.posteriorParamDict['latentPhenotypeEffects']['rate']).to_event(1))
        else:
            self.posteriorParamDict['latentPhenotypeEffects']['mean']=pyro.param('latentPhenotypeEffectsPosteriorConc',init_tensor=self.posteriorParamDict['latentPhenotypeEffects']['mean'])
            self.posteriorParamDict['latentPhenotypeEffects']['scale']=pyro.param('latentPhenotypeEffectsPosteriorScale',init_tensor=self.posteriorParamDict['latentPhenotypeEffects']['scale'],constraint=torch.distributions.constraints.positive)
            with pyro.plate("latent_pheno_effect_plate",size=self.nLatentDim,dim=-1):
                pyro.sample("latentPhenotypeEffects",dist.Normal(self.posteriorParamDict['latentPhenotypeEffects']['mean'],self.posteriorParamDict['latentPhenotypeEffects']['scale']).to_event(1))
 
        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                z_mean,z_std = self.encoder(obs_data,*cat_cov_list)
                pyro.sample("latentPhenotypes", dist.Normal(z_mean, z_std).to_event(1))

    def ComputeELBO(self,obs_dis_array,cat_cov_list,num_particles=10):
        elboFunc = Trace_ELBO(num_particles=num_particles)
        return elboFunc.loss()

    def ComputeELBOPerDatum(self,obs_dis_array,cat_cov_list,num_particles):
        elboFunc = Trace_ELBO(num_particles=num_particles)
        elboVec = torch.zeros(obs_dis_array.shape[0],dtype=torch.float32,device=self.compute_device)

        for model_trace, guide_trace in elboFunc._get_traces(self.model, self.guide,obs_dis_array,cat_cov_list):
            elboVec+=model_trace.nodes['obsTraitIncidence']['log_prob'].detach()/num_particles
            elboVec+=model_trace.nodes['latentPhenotypes']['log_prob'].detach()/num_particles
            elboVec-=guide_trace.nodes['latentPhenotypes']['log_prob'].detach()/num_particles
        return elboVec.reshape(elboVec.shape[0],1)

    def PredictLatentPhenotypes(self,obs_dis_array,cat_cov_list,returnScale=False,num_particles=10):
        z_mean,z_scale = self.encoder(obs_dis_array,*cat_cov_list)
        if returnScale:
            return z_mean,z_scale
        else:
            return z_mean

    def ComputeKLDivergenceFromPrior(self,obs_dis_array,cat_cov_list,returnScale=False,num_particles=10):

        distParams = self.PredictLatentPhenotype(obs_dis_array,cat_cov_list,returnScale=True,num_particles=num_particles)
        return 0.5*torch.sum((distParams[0]**2)+(distParams[1]**2)-2.0*torch.log(distParams[1])-1.0,dim=1,keepdim=True)



    def PackageCurrentState(self):
        packaged_model_state={}
        packaged_model_state['model_state'] = copy.deepcopy(self.state_dict(keep_vars=True))
        packaged_model_state['posterior_params'] = self._copyModelPosteriorParams()
        packaged_model_state['prior_params'] = self._copyModelPriorParams()
        return packaged_model_state

    def LoadPriorState(self,prior_model_state):
        self.load_state_dict(prior_model_state['model_state'],strict=True)
        self.posteriorParamDict=prior_model_state['posterior_params']
        self.priorParamDict=prior_model_state['prior_params']


if __name__=='__main__':
    
    from vlpi.ClinicalDataSimulator import ClinicalDataSimulator
    from vlpi.ClinicalDataset import ClinicalDatasetSampler,ClinicalDataset 
    
    
    from pyro import poutine
    pyro.enable_validation(True)    # <---- This is always a good idea!

    # We'll ue this helper to check our models are correct.
    def test_model(model, guide, loss):
        pyro.clear_param_store()
        loss.loss(model, guide)
        
        
    numSamples = 100
    numAssociatedTraits=20
    nLatentSimDim=4
    nLatentFitDim=4
    mappingFunction='Linear'
    numCovPerClass = [2,3,10] 
    covNames = ['A','B','C']

    
    
    simulator = ClinicalDataSimulator(numAssociatedTraits,nLatentSimDim,numCatList=numCovPerClass)
    simData=simulator.GenerateClinicalData(numSamples,0.0)
    
    clinData = ClinicalDataset()
    
    disList = list(clinData.dxCodeToDataIndexMap.keys())[0:numAssociatedTraits]
    clinData.IncludeOnly(disList)
    
        
        
        
    clinData.LoadFromArrays(simData['incidence_data'],simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')
    sampler = ClinicalDatasetSampler(clinData,0.75,returnArrays='Torch')
    
    _vlpi=_vlpiModel(numAssociatedTraits, numCovPerClass,nLatentSimDim,mappingFunction=mappingFunction)
    
    trace = poutine.trace(_vlpi.model).get_trace()
    
    trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    print(trace.format_shapes())
    
    def model():
        return _vlpi.model(*sampler.ReturnFullTrainingDataset(randomize=False))
    
    def guide():
        return _vlpi.guide(*sampler.ReturnFullTrainingDataset(randomize=False))
    
    test_model(model,guide,loss=Trace_ELBO())