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
from vlpi.utils import random_catcov,infer_liability_CI,build_onehot_arrays

import copy

class ContinuousModel(nn.Module):



    def _computeCovEffects(self,cat_cov_list,cov_effect_array):
        one_hot_cat_list = build_onehot_arrays(cat_cov_list,self.numCatList,self.dropLinearCovariateColumn)
        return torch.mm(torch.cat((*one_hot_cat_list,),dim=-1),cov_effect_array)


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
    
    def _encoder_only_model(self,obs_data=None, cat_cov_list=None, anchor_dx=None,sample_scores=None,numSamples=None,minibatch_scale=1.0):
        assert sample_scores is not None, "Sample scores must be included in order to train the encoder individually."
        assert obs_data is not None, "obs_data must be included when using the _encoder_only_model."
        pyro.module("encoder", self.encoder)
            
        if self.useAnchorDx > 0:
            assert anchor_dx is not None,"Model requires input of anchor dx data if marked as included."
            assert anchor_dx.shape[0]==obs_data.shape[0],"Dimensions along axis 0 of anchor and observed dx data must match."
            obs_data = torch.cat((obs_data,anchor_dx),dim=-1)
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
 
        
    def _encoder_only_guide(self,obs_data=None, cat_cov_list=None, anchor_dx=None,sample_scores=None,numSamples=None,minibatch_scale=1.0):
        assert sample_scores is not None, "Sample scores must be included in order to train the encoder individually."
        assert obs_data is not None, "obs_data must be included when using the _encoder_only_model."
            
        if self.useAnchorDx > 0:
            assert anchor_dx is not None,"Model requires input of anchor dx data if marked as included."
            assert anchor_dx.shape[0]==obs_data.shape[0],"Dimensions along axis 0 of anchor and observed dx data must match."


    def __init__(self,numObsTraits:int, numCatList:Iterable[int],useAnchorDx:bool,nLatentDim:int,mappingFunction:str,anchorDxPriors={'anchorDxNoise':[1.0,1.0],'latentDimToAnchorDxMap':1.0,'prevalence':[-3.0,3.0]},latentPhenotypePriors={'element_wise_precision':[1.0,1.0]},covariatePriors={'intercept':[0.0,5.0],'cov_scale':3.0},**kwargs):




        super(ContinuousModel,self).__init__()
        self.numObsTraits=numObsTraits
        self.numCatList=numCatList
        self.useAnchorDx=useAnchorDx
        self.nLatentDim = nLatentDim


        assert mappingFunction in ['Linear','Linear_Monotonic'], "Currently supported mapping model types for vLPM: Linear, Linear_Monotonic"

        self.mappingFunction=mappingFunction

        allKeywordArgs = list(kwargs.keys())
        
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
            self.encoderHyperparameters={'n_layers' : 2, 'n_hidden' : 32, 'dropout_rate': 0.1, 'use_batch_norm':True}

        else:
            self.encoderHyperparameters = kwargs['neuralNetworkHyperparameters']
            assert isinstance(self.encoderHyperparameters,dict),"Expects dictionary of encoder hyperparameters"
            assert set(self.encoderHyperparameters.keys())==set(['n_layers','n_hidden','dropout_rate','use_batch_norm']),"Encoder hyperparameters must include: 'n_layers','n_hidden','dropout_rate','use_batch_norm'"

        if self.dropLinearCovariateColumn:
            self.numCovParam = sum(self.numCatList)-len(self.numCatList)
        else:
            self.numCovParam = sum(self.numCatList)

        self.encoder=MeanScaleEncoder(self.numObsTraits+int(self.useAnchorDx), self.nLatentDim, n_cat_list= self.numCatList, **self.encoderHyperparameters)

        self.posteriorParamDict = {}
        self.priorParamDict={}


        self.priorParamDict['intercepts']={}
        self.priorParamDict['intercepts']['mean'] = torch.tensor(covariatePriors['intercept'][0],dtype=torch.float32)
        self.priorParamDict['intercepts']['scale'] = torch.tensor(covariatePriors['intercept'][1],dtype=torch.float32)
        self.posteriorParamDict['intercepts'] = {'mean':torch.ones(self.numObsTraits,dtype=torch.float32)*covariatePriors['intercept'][0],'scale':torch.ones(self.numObsTraits,dtype=torch.float32)*covariatePriors['intercept'][1]}


        if self.numCovParam>0:
            self.posteriorParamDict['covEffects']={'mean':torch.zeros(self.numCovParam,self.numObsTraits,dtype=torch.float32),'scale':torch.ones(self.numCovParam,self.numObsTraits,dtype=torch.float32)*covariatePriors['cov_scale']}
            self.priorParamDict['covEffects']={}
            self.priorParamDict['covEffects']['mean'] = torch.tensor(0.0,dtype=torch.float32)
            self.priorParamDict['covEffects']['scale'] = torch.tensor(covariatePriors['cov_scale'],dtype=torch.float32)
            

        
        self.priorParamDict['latentPhenotypeEffectsPrecision']={}
        self.priorParamDict['latentPhenotypeEffectsPrecision']['conc']=torch.tensor(latentPhenotypePriors['element_wise_precision'][0],dtype=torch.float32)
        self.priorParamDict['latentPhenotypeEffectsPrecision']['rate']=torch.tensor(latentPhenotypePriors['element_wise_precision'][1],dtype=torch.float32)
        self.posteriorParamDict['latentPhenotypeEffectsPrecision']= {'conc':torch.ones(self.nLatentDim,1,dtype=torch.float32)*latentPhenotypePriors['element_wise_precision'][0],'rate':torch.ones(self.nLatentDim,1,dtype=torch.float32)*latentPhenotypePriors['element_wise_precision'][1]}

        if self.useAnchorDx:

            self.priorParamDict['anchorDxNoise']={}
            self.priorParamDict['anchorDxNoise']['alpha']=torch.tensor(anchorDxPriors['anchorDxNoise'][0],dtype=torch.float32)
            self.priorParamDict['anchorDxNoise']['beta']=torch.tensor(anchorDxPriors['anchorDxNoise'][1],dtype=torch.float32)
            self.posteriorParamDict['anchorDxNoise'] = {'alpha':torch.tensor([anchorDxPriors['anchorDxNoise'][0]],dtype=torch.float32),'beta':torch.tensor([anchorDxPriors['anchorDxNoise'][1]],dtype=torch.float32)}
            
            transformed_latent_pheno_prior=infer_liability_CI(anchorDxPriors['prevalence'])
            self.priorParamDict['latentPhenotypePrevalence']={}
            self.priorParamDict['latentPhenotypePrevalence']['mean']=torch.tensor(transformed_latent_pheno_prior[0],dtype=torch.float32)
            self.priorParamDict['latentPhenotypePrevalence']['scale']=torch.tensor(transformed_latent_pheno_prior[1],dtype=torch.float32)
            self.posteriorParamDict['latentPhenotypePrevalence'] = {'mean':self.priorParamDict['latentPhenotypePrevalence']['mean'].detach(),'scale':self.priorParamDict['latentPhenotypePrevalence']['scale'].detach()}
            if self.nLatentDim>1:
                self.priorParamDict['latentDimToAnchorMap']={}
                self.priorParamDict['latentDimToAnchorMap']['alpha']=(anchorDxPriors['latentDimToAnchorDxMap']/self.nLatentDim)*torch.ones(self.nLatentDim,dtype=torch.float32)
                self.posteriorParamDict['latentDimToAnchorMap'] = {'alpha':torch.ones(self.nLatentDim,dtype=torch.float32)}



        if self.mappingFunction=='Linear_Monotonic':
            self.posteriorParamDict['latentPhenotypeEffects'] = {'conc':torch.ones(self.numObsTraits,dtype=torch.float32),'rate':torch.ones(self.nLatentDim,self.numObsTraits,dtype=torch.float32)}
        else:
            self.posteriorParamDict['latentPhenotypeEffects'] = {'mean':torch.zeros(self.nLatentDim,self.numObsTraits,dtype=torch.float32),'scale':torch.ones(self.nLatentDim,self.numObsTraits,dtype=torch.float32)}


        if self.compute_device is not None:
            self.SwitchDevice(self.compute_device)

        self.eval()

    def SimulateData(self,numSamples):
        return self.model(numSamples=numSamples)
    
    def model(self, obs_data=None, cat_cov_list=None, anchor_dx=None,sample_scores=None,numSamples=None,minibatch_scale=1.0):
        if obs_data is not None:
            numSamples=obs_data.shape[0]
            if self.useAnchorDx > 0:
                assert anchor_dx is not None,"Model requires input of anchor dx data if marked as included."
                assert anchor_dx.shape[0]==obs_data.shape[0],"Dimensions along axis 0 of anchor and observed dx data must match."
                obs_data = torch.cat((obs_data,anchor_dx),dim=-1)
    
    
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

        if self.useAnchorDx:
            anchorDxNoise = pyro.sample("anchorDxNoise",dist.Beta(self.priorParamDict['anchorDxNoise']['alpha'],self.priorParamDict['anchorDxNoise']['beta']))
            lp_prev_liability = pyro.sample("latentPhenotypePrevalence",dist.Normal(self.priorParamDict['latentPhenotypePrevalence']['mean'],self.priorParamDict['latentPhenotypePrevalence']['scale']))
            if self.nLatentDim > 1:
                latentDimToAnchorMap = pyro.sample("latentDimToAnchorMap",dist.Dirichlet(self.priorParamDict['latentDimToAnchorMap']['alpha']))

        if self.numCovParam>0:
            with pyro.plate("cov_plate",size=self.numCovParam):
                covEffects = pyro.sample("covEffects",dist.Normal(self.priorParamDict['covEffects']['mean'],self.priorParamDict['covEffects']['scale']).expand([self.numObsTraits]).to_event(1))
        
        with pyro.plate("loading_prec_plate",size=self.nLatentDim,dim=-1):
            latentPhenotypePrecision = pyro.sample("latentPhenotypeEffectsPrecision",dist.Gamma(self.priorParamDict['latentPhenotypeEffectsPrecision']['conc'],self.priorParamDict['latentPhenotypeEffectsPrecision']['rate']).expand([1]).to_event(1))
        
        if self.mappingFunction=='Linear_Monotonic':
            with pyro.plate("latent_pheno_effect_plate",size=self.nLatentDim,dim=-1):
                latentPhenotypeEffects = pyro.sample("latentPhenotypeEffects",dist.Exponential(torch.sqrt(latentPhenotypePrecision)).expand([self.nLatentDim,self.numObsTraits]).to_event(1))
        else:
            with pyro.plate("latent_pheno_effect_plate",size=self.numObsTraits,dim=-1):
                latentPhenotypeEffects = pyro.sample("latentPhenotypeEffects",dist.Normal(0.0,torch.sqrt(1.0/latentPhenotypePrecision)).expand([self.nLatentDim,self.numObsTraits]).to_event(1))


        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                latentPhenotypes=pyro.sample("latentPhenotypes",dist.Normal(torch.zeros(1,self.nLatentDim,dtype=torch.float32,device=self.compute_device),torch.ones(1,self.nLatentDim,dtype=torch.float32,device=self.compute_device)).to_event(1))
                liabilities = torch.mm(latentPhenotypes,latentPhenotypeEffects)+intercepts
                if self.numCovParam>0:
                    liabilities+=self._computeCovEffects(cat_cov_list,covEffects)
                mean_dx_rates=dist.Normal(0.0,1.0).cdf(liabilities)
                
                
                if self.useAnchorDx:
                    if self.nLatentDim > 1:
                        anchor_dx_liability = (torch.sum(latentPhenotypes*torch.sqrt(latentDimToAnchorMap),dim=-1,keepdim=True)+lp_prev_liability)/anchorDxNoise
                    else:
                        anchor_dx_liability=(latentPhenotypes+lp_prev_liability)/anchorDxNoise
                    anchor_dx_prob = dist.Normal(0.0,1.0).cdf(anchor_dx_liability)
                    
                    mean_dx_rates = torch.cat((mean_dx_rates,anchor_dx_prob),dim=-1)
                sample_results=pyro.sample("obsTraitIncidence",dist.Bernoulli(mean_dx_rates).to_event(1),obs=obs_data)
        if obs_data is None:
            outputDict ={}
            outputDict['model_params']={}
            outputDict['model_params']['intercepts']=intercepts
            outputDict['model_params']['latentPhenotypeEffectsPrecision']=latentPhenotypePrecision
            outputDict['model_params']['latentPhenotypeEffects']=latentPhenotypeEffects

            outputDict['incidence_data']=sample_results[:,:self.numObsTraits]
            outputDict['latent_phenotypes']=latentPhenotypes
            if self.numCovParam>0:
                outputDict['covariate_data']=cat_cov_list
                outputDict['model_params']['covEffects']=covEffects
            else:
                outputDict['covariate_data']=[]
            if self.useAnchorDx:
                outputDict['anchor_dx_data'] = sample_results[:,self.numObsTraits:(self.numObsTraits+1)]
                outputDict['model_params']['anchorDxNoise']=anchorDxNoise
                outputDict['model_params']['latentPhenotypePrevalence']=lp_prev_liability
                if self.nLatentDim > 1:
                    outputDict['model_params']['latentDimToAnchorMap']=latentDimToAnchorMap
            return outputDict

            
            

    def guide(self,obs_data=None, cat_cov_list=None,anchor_dx=None,sample_scores=None,numSamples = None,minibatch_scale=1.0):
        if obs_data is not None:
            numSamples=obs_data.shape[0]

            if self.useAnchorDx > 0:
                assert anchor_dx is not None,"Model requires input of anchor dx data if marked as included."
                assert anchor_dx.shape[0]==obs_data.shape[0],"Dimensions along axis 0 of anchor and observed dx data must match."
                obs_data = torch.cat((obs_data,anchor_dx),dim=-1)
        else:
            pass
        
        if sample_scores is None:
            pyro.module("encoder", self.encoder)
            

        self.posteriorParamDict['intercepts']['mean']=pyro.param("interceptPosteriorMean",init_tensor=self.posteriorParamDict['intercepts']['mean'])
        self.posteriorParamDict['intercepts']['scale']=pyro.param("interceptPosteriorScale",init_tensor=self.posteriorParamDict['intercepts']['scale'],constraint=torch.distributions.constraints.positive)
        with pyro.plate("intercept_plate",size=self.numObsTraits,dim=-1):
            pyro.sample("intercepts",dist.Normal(self.posteriorParamDict['intercepts']['mean'] ,self.posteriorParamDict['intercepts']['scale']))
        
        self.posteriorParamDict['latentPhenotypeEffectsPrecision']['conc']=pyro.param('latentPhenotypeEffectsPrecisionPosteriorConc',init_tensor=self.posteriorParamDict['latentPhenotypeEffectsPrecision']['conc'],constraint=torch.distributions.constraints.positive)
        self.posteriorParamDict['latentPhenotypeEffectsPrecision']['rate']=pyro.param('latentPhenotypeEffectsPrecisionPosteriorRate',init_tensor=self.posteriorParamDict['latentPhenotypeEffectsPrecision']['rate'],constraint=torch.distributions.constraints.positive)
        with pyro.plate("loading_prec_plate",size=self.nLatentDim,dim=-1):
            pyro.sample("latentPhenotypeEffectsPrecision",dist.Gamma(self.posteriorParamDict['latentPhenotypeEffectsPrecision']['conc'],self.posteriorParamDict['latentPhenotypeEffectsPrecision']['rate']).to_event(1))
        

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



        if self.numCovParam>0:
            self.posteriorParamDict['covEffects']['mean']=pyro.param("covEffectsPosteriorMean",init_tensor=self.posteriorParamDict['covEffects']['mean'])
            self.posteriorParamDict['covEffects']['scale']=pyro.param("covEffectsPosteriorScale",init_tensor=self.posteriorParamDict['covEffects']['scale'],constraint=torch.distributions.constraints.positive)
            with pyro.plate("cov_plate",size=self.numCovParam):
                pyro.sample("covEffects",dist.Normal(self.posteriorParamDict['covEffects']['mean'],self.posteriorParamDict['covEffects']['scale']).to_event(1))


        if self.useAnchorDx > 0:

            self.posteriorParamDict['anchorDxNoise']['alpha']=pyro.param('anchorDxNoisePosteriorAlpha',init_tensor=self.posteriorParamDict['anchorDxNoise']['alpha'],constraint=torch.distributions.constraints.positive)
            self.posteriorParamDict['anchorDxNoise']['beta']=pyro.param('anchorDxNoisePosteriorBeta',init_tensor=self.posteriorParamDict['anchorDxNoise']['beta'],constraint=torch.distributions.constraints.positive)
            pyro.sample('anchorDxNoise',dist.Beta(self.posteriorParamDict['anchorDxNoise']['alpha'],self.posteriorParamDict['anchorDxNoise']['beta']))
            
            self.posteriorParamDict['latentPhenotypePrevalence']['mean']=pyro.param('latentPhenotypePrevalencePosteriorMean',init_tensor=self.posteriorParamDict['latentPhenotypePrevalence']['mean'])
            self.posteriorParamDict['latentPhenotypePrevalence']['scale']=pyro.param('latentPhenotypePrevalencePosteriorScale',init_tensor=self.posteriorParamDict['latentPhenotypePrevalence']['scale'],constraint=torch.distributions.constraints.positive)
            pyro.sample("latentPhenotypePrevalence",dist.Normal(self.posteriorParamDict['latentPhenotypePrevalence']['mean'],self.posteriorParamDict['latentPhenotypePrevalence']['scale']))
            if self.nLatentDim > 1:
                self.posteriorParamDict['latentDimToAnchorMap']['alpha']=pyro.param('latentDimToAnchorMapPosterior',self.posteriorParamDict['latentDimToAnchorMap']['alpha'],constraint=torch.distributions.constraints.positive)
                pyro.sample('latentDimToAnchorMap',dist.Dirichlet(self.posteriorParamDict['latentDimToAnchorMap']['alpha']))
        

        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                z_mean,z_std = self.encoder(obs_data,*cat_cov_list)
                pyro.sample("latentPhenotypes", dist.Normal(z_mean, z_std).to_event(1))



    def _computeELBOPerDatum(self,obs_dis_array,cat_cov_list,anchor_dx_array,num_particles):
        elboFunc = Trace_ELBO(num_particles=num_particles)
        elboVec = torch.zeros(anchor_dx_array.shape[0],dtype=torch.float32,device=self.compute_device)

        for model_trace, guide_trace in elboFunc._get_traces(self.model, self.guide,obs_dis_array,cat_cov_list,anchor_dx_array):
            elboVec+=model_trace.nodes['obsTraitIncidence']['log_prob'].detach()/num_particles
            elboVec+=model_trace.nodes['latentPhenotypes']['log_prob'].detach()/num_particles
            elboVec-=guide_trace.nodes['latentPhenotypes']['log_prob'].detach()/num_particles
        return elboVec.reshape(elboVec.shape[0],1)

    def PredictLatentPhenotypes(self,obs_dis_array,cat_cov_list,anchor_dx_array=None,returnScale=False,num_particles=10,anchor_dx_prior = 0.5):
        if self.useAnchorDx:
            if anchor_dx_array is not None:
                output = self.encoder(torch.cat((obs_dis_array,anchor_dx_array),dim=-1),*cat_cov_list)
                z_mean,z_scale=output
            else:
                if not torch.is_tensor(anchor_dx_prior):
                    anchor_dx_prior=torch.tensor(anchor_dx_prior,dtype=torch.float32,device =obs_dis_array.device)
                assert (anchor_dx_prior.item()<1.0) and (anchor_dx_prior.item()>0.0),"anchor dx prior must be between 0 and 1"
                elboWDx = self._computeELBOPerDatum(obs_dis_array,cat_cov_list,torch.ones((obs_dis_array.shape[0],1)),num_particles)
                elboWoDx = self._computeELBOPerDatum(obs_dis_array,cat_cov_list,torch.zeros((obs_dis_array.shape[0],1)),num_particles)
                probDx = torch.exp(elboWDx+torch.log(anchor_dx_prior)-torch.logsumexp(torch.cat((elboWDx+torch.log(anchor_dx_prior),elboWoDx+torch.log(1.0-anchor_dx_prior)),dim=1),dim=1,keepdim=True))

                output_on = self.encoder(torch.cat((obs_dis_array,torch.ones((obs_dis_array.shape[0],1))),dim=-1),*cat_cov_list)
                output_off = self.encoder(torch.cat((obs_dis_array,torch.zeros((obs_dis_array.shape[0],1))),dim=-1),*cat_cov_list)

                z_mean = probDx*output_on[0]+(1.0-probDx)*output_off[0]
                z_scale = torch.sqrt(probDx*(output_on[0]**2.0+output_on[1]**2.0-z_mean**2)+(1.0-probDx)*(output_off[0]**2.0+output_off[1]**2.0-z_mean**2))

            if self.nLatentDim > 1:
                expLatentToAnchorMap = self.posteriorParamDict['latentDimToAnchorMap']['alpha']/torch.sum(self.posteriorParamDict['latentDimToAnchorMap']['alpha'])
                z_mean_transformed=torch.sum(torch.sqrt(expLatentToAnchorMap)*z_mean,dim=-1,keepdim=True)
                z_scale_transformed = torch.sqrt(torch.sum((expLatentToAnchorMap*z_scale)**2,dim=-1,keepdim=True))
                
                if returnScale:
                    return (z_mean,z_scale),(z_mean_transformed,z_scale_transformed)
                else:
                    return (z_mean,z_mean_transformed)
            else:
                if returnScale:
                    return z_mean,z_scale
                else:
                    return z_mean

        else:
            z_mean,z_scale = self.encoder(obs_dis_array,*cat_cov_list)

            if returnScale:
               return z_mean,z_scale
            else:
               return z_mean

    def ComputeKLDivergenceFromPrior(self,obs_dis_array,cat_cov_list,anchor_dx_array=None,returnScale=False,num_particles=5,anchor_dx_prior = 0.5):
        
        distParams = self.PredictLatentPhenotype(obs_dis_array,cat_cov_list,anchor_dx_array=anchor_dx_array,returnScale=True,num_particles=num_particles,anchor_dx_prior = anchor_dx_prior)
        return 0.5*torch.sum((distParams[0]**2)+(distParams[1]**2)-2.0*torch.log(distParams[1])-1.0,dim=1,keepdim=True)
    
    
    def PredictAnchorDx(self,obs_dis_array,cat_cov_list,num_particles=5,anchor_dx_prior = 0.5):
        assert self.useAnchorDx, "Cannot predict anchor dx if model not trained using this information"
        if not torch.is_tensor(anchor_dx_prior):
            anchor_dx_prior=torch.tensor(anchor_dx_prior,dtype=torch.float32,device =obs_dis_array.device)
        assert anchor_dx_prior.item()<1.0 and anchor_dx_prior.item()>0.0,"anchor dx prior must be between 0 and 1"
        elboWDx = self._computeELBOPerDatum(obs_dis_array,cat_cov_list,torch.ones((obs_dis_array.shape[0],1)),num_particles)
        elboWoDx = self._computeELBOPerDatum(obs_dis_array,cat_cov_list,torch.zeros((obs_dis_array.shape[0],1)),num_particles)
        probDx = torch.exp(elboWDx+torch.log(anchor_dx_prior)-torch.logsumexp(torch.cat((elboWDx+torch.log(anchor_dx_prior),elboWoDx+torch.log(1.0-anchor_dx_prior)),dim=1),dim=1,keepdim=True))
        return probDx



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






