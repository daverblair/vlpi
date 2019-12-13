#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:13:38 2019

@author: davidblair
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate,TraceEnum_ELBO
from vlpi.Encoders import BinaryExpectationEncoder
from typing import Iterable
from vlpi.utils import random_catcov,build_onehot_arrays
import copy

class DiscreteModel(nn.Module):

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
        
        rankings=sample_scores.argsort(dim=0,descending=True)
#        rank_cutoff=int(sample_scores.shape[0]*dist.Normal(0.0,1.0).cdf(self.priorParamDict['latentPhenotypePrevalence']['mean']))
        rank_cutoff=int(sample_scores.shape[0]*(self.priorParamDict['latentPhenotypePrevalence']['alpha']/(self.priorParamDict['latentPhenotypePrevalence']['alpha']+self.priorParamDict['latentPhenotypePrevalence']['beta'])))
        phenotype_vals = torch.zeros(sample_scores.shape,dtype=torch.long,device=sample_scores.device)
        phenotype_vals[rankings[0:rank_cutoff]]=1
        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("rank_pheno_plate",size=numSamples):
                latent_pheno_post = self.encoder(obs_data,*cat_cov_list)
                pyro.sample("rankPhenotypes", dist.Categorical(probs=latent_pheno_post),obs=phenotype_vals.flatten())
 
        
    def _encoder_only_guide(self,obs_data=None, cat_cov_list=None, anchor_dx=None,sample_scores=None,numSamples=None,minibatch_scale=1.0):
        assert sample_scores is not None, "Sample scores must be included in order to train the encoder individually."
        assert obs_data is not None, "obs_data must be included when using the _encoder_only_model."
            
        if self.useAnchorDx > 0:
            assert anchor_dx is not None,"Model requires input of anchor dx data if marked as included."
            assert anchor_dx.shape[0]==obs_data.shape[0],"Dimensions along axis 0 of anchor and observed dx data must match."
        
    
    def __init__(self, numObsTraits:int, numCatList:Iterable[int],useAnchorDx:bool,mappingFunction:str,anchorDxPriors={'anchorDxNoise':[1.0,1.0]},latentPhenotypePriors={'element_wise_precision':[1.0,1.0],'prevalence':[-3.0,3.0]},covariatePriors={'intercept':[0.0,5.0],'cov_scale':3.0},**kwargs):

        super(DiscreteModel,self).__init__()
        self.numObsTraits=numObsTraits
        self.numCatList=numCatList
        self.useAnchorDx=useAnchorDx
        


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

        if self.dropLinearCovariateColumn:
            self.numCovParam = sum(self.numCatList)-len(self.numCatList)
        else:
            self.numCovParam = sum(self.numCatList)

        if 'neuralNetworkHyperparameters' not in allKeywordArgs:
            self.encoderHyperparameters={'n_layers' : 2, 'n_hidden' : 128, 'dropout_rate': 0.2, 'use_batch_norm':True}

        else:
            self.encoderHyperparameters = kwargs['neuralNetworkHyperparameters']
            assert isinstance(self.encoderHyperparameters,dict),"Expects dictionary of encoder hyperparameters"
            assert set(self.encoderHyperparameters.keys())==set(['n_layers','n_hidden','dropout_rate','use_batch_norm']),"Encoder hyperparameters must include: 'n_layers','n_hidden','dropout_rate','use_batch_norm'"
        
        

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

        self.priorParamDict['latentPhenotypePrevalence']={}
        self.priorParamDict['latentPhenotypePrevalence']['alpha']=torch.tensor([latentPhenotypePriors['prevalence'][0]],dtype=torch.float32)
        self.priorParamDict['latentPhenotypePrevalence']['beta']=torch.tensor([latentPhenotypePriors['prevalence'][1]],dtype=torch.float32)
        self.posteriorParamDict['latentPhenotypePrevalence'] = {'alpha':torch.tensor([latentPhenotypePriors['prevalence'][0]],dtype=torch.float32),'beta':torch.tensor([latentPhenotypePriors['prevalence'][1]],dtype=torch.float32)}

        self.priorParamDict['latentPhenotypeEffectsPrecision']={}
        self.priorParamDict['latentPhenotypeEffectsPrecision']['conc']=torch.tensor(latentPhenotypePriors['element_wise_precision'][0],dtype=torch.float32)
        self.priorParamDict['latentPhenotypeEffectsPrecision']['rate']=torch.tensor(latentPhenotypePriors['element_wise_precision'][1],dtype=torch.float32)
        self.posteriorParamDict['latentPhenotypeEffectsPrecision']= {'conc':torch.tensor([latentPhenotypePriors['element_wise_precision'][0]],dtype=torch.float32),'rate':torch.tensor([latentPhenotypePriors['element_wise_precision'][1]],dtype=torch.float32)}

        if self.useAnchorDx:

            self.priorParamDict['anchorDxNoise']={}
            self.priorParamDict['anchorDxNoise']['conc']=torch.tensor(anchorDxPriors['anchorDxNoise'][0],dtype=torch.float32)
            self.priorParamDict['anchorDxNoise']['rate']=torch.tensor(anchorDxPriors['anchorDxNoise'][1],dtype=torch.float32)
            self.posteriorParamDict['anchorDxNoise'] = {'conc':torch.tensor([anchorDxPriors['anchorDxNoise'][0]],dtype=torch.float32),'rate':torch.tensor([anchorDxPriors['anchorDxNoise'][1]],dtype=torch.float32)}


        if self.mappingFunction=='Linear_Monotonic':
            self.posteriorParamDict['latentPhenotypeEffects'] = {'conc':torch.ones(1,self.numObsTraits,dtype=torch.float32),'rate':torch.ones(1,self.numObsTraits,dtype=torch.float32)}
        else:
            self.posteriorParamDict['latentPhenotypeEffects'] = {'mean':torch.zeros(1,self.numObsTraits,dtype=torch.float32),'scale':torch.ones(1,self.numObsTraits,dtype=torch.float32)}


        self.encoder=BinaryExpectationEncoder(self.numObsTraits+int(self.useAnchorDx), 1, n_cat_list=self.numCatList, **self.encoderHyperparameters)
        if self.compute_device is not None:
            self.SwitchDevice(self.compute_device)
        self.eval()

    def SimulateData(self,numSamples):
        return self.model(numSamples=numSamples)
    
    @config_enumerate
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
                print('Warning: no arguments were given to vlPIDiscrete.model. This should only be done during debugging.')
            cat_cov_list = []
            for n_cat in self.numCatList:
                cat_cov_list+=[random_catcov(n_cat,numSamples,device=self.compute_device)]

        with pyro.plate("intercept_plate",size=self.numObsTraits,dim=-1):
            intercepts = pyro.sample("intercepts",dist.Normal(self.priorParamDict['intercepts']['mean'],self.priorParamDict['intercepts']['scale']))

        if self.useAnchorDx:
            anchorDxNoise = pyro.sample("anchorDxNoise",dist.Gamma(self.priorParamDict['anchorDxNoise']['conc'],self.priorParamDict['anchorDxNoise']['rate']))

        if self.numCovParam>0:
            with pyro.plate("cov_plate",size=self.numCovParam):
                covEffects = pyro.sample("covEffects",dist.Normal(self.priorParamDict['covEffects']['mean'],self.priorParamDict['covEffects']['scale']).expand([self.numObsTraits]).to_event(1))

        latentPhenotypePrecision = pyro.sample("latentPhenotypeEffectsPrecision",dist.Gamma(self.priorParamDict['latentPhenotypeEffectsPrecision']['conc'],self.priorParamDict['latentPhenotypeEffectsPrecision']['rate']))


        if self.mappingFunction=='Linear_Monotonic':
            latentPhenotypeEffects = pyro.sample("latentPhenotypeEffects",dist.Exponential(torch.sqrt(latentPhenotypePrecision)).expand([1,self.numObsTraits]).to_event(1))
        else:
            latentPhenotypeEffects = pyro.sample("latentPhenotypeEffects",dist.Normal(0.0,torch.sqrt(1.0/latentPhenotypePrecision)).expand([1,self.numObsTraits]).to_event(1))


        lp_prev = pyro.sample("latentPhenotypePrevalence",dist.Beta(self.priorParamDict['latentPhenotypePrevalence']['alpha'],self.priorParamDict['latentPhenotypePrevalence']['beta']))
#        lp_prev=dist.Normal(0.0,1.0).cdf(lp_prev_liability)
        dx_rates = torch.zeros(2,self.numObsTraits,dtype=torch.float32,device=self.compute_device)
        dx_rates[1]=intercepts+latentPhenotypeEffects
        dx_rates[0]=intercepts

        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                latentPhenotypes=pyro.sample("latentPhenotypes",dist.Categorical(probs=torch.tensor([1.0-lp_prev,lp_prev],dtype=torch.float32,device=self.compute_device)))
                
                mean_dx_rates = dx_rates[latentPhenotypes]
                
                if self.numCovParam>0:
                    mean_dx_rates=mean_dx_rates+self._computeCovEffects(cat_cov_list,covEffects)
                mean_dx_rates=dist.Normal(0.0,1.0).cdf(mean_dx_rates)
                
                if self.useAnchorDx:
                    tmp=torch.zeros(2,dtype=torch.float32,device=latentPhenotypes.device)
                    tmp[1]=torch.sigmoid(1.0/anchorDxNoise)
                    tmp[0]=1.0-tmp[1]
                    anchor_dx_prob = tmp[latentPhenotypes]
                    if len(mean_dx_rates.shape)==2:
                        mean_dx_rates = torch.cat((mean_dx_rates,anchor_dx_prob.reshape((mean_dx_rates.shape[-2],1))),dim=-1)
                    else:
                        anchor_dx_prob=anchor_dx_prob.unsqueeze(1)
                        mean_dx_rates = torch.cat((mean_dx_rates,anchor_dx_prob*torch.ones((2,mean_dx_rates.shape[1],1),dtype=torch.float32,device=self.compute_device)),dim=-1)
                sample_results = pyro.sample("obsTraitIncidence",dist.Bernoulli(mean_dx_rates).to_event(1),obs=obs_data)
        if obs_data is None:
            outputDict ={}
            outputDict['model_params']={}
            outputDict['model_params']['intercepts']=intercepts
            outputDict['model_params']['latentPhenotypePrevalence']=lp_prev
            outputDict['model_params']['latentPhenotypeEffectsPrecision']=latentPhenotypePrecision
            outputDict['model_params']['latentPhenotypeEffects']=latentPhenotypeEffects

            outputDict['incidence_data']=sample_results[:,:self.numObsTraits]
            outputDict['latent_phenotypes']=latentPhenotypes.unsqueeze(-1)
            if self.numCovParam>0:
                outputDict['covariate_data']=cat_cov_list
                outputDict['model_params']['covEffects']=covEffects
            else:
                outputDict['covariate_data']=[]
            if self.useAnchorDx:
                outputDict['anchor_dx_data'] = sample_results[:,self.numObsTraits:(self.numObsTraits+1)]
                outputDict['model_params']['anchorDxNoise']=anchorDxNoise
            return outputDict


    @config_enumerate
    def guide(self,obs_data=None, cat_cov_list=None, anchor_dx=None,sample_scores=None,numSamples = None,minibatch_scale=1.0):
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

        self.posteriorParamDict['intercepts']['mean']=pyro.param("interceptPosteriorMean",init_tensor=self.posteriorParamDict['intercepts']['mean'].detach())
        self.posteriorParamDict['intercepts']['scale']=pyro.param("interceptPosteriorScale",init_tensor=self.posteriorParamDict['intercepts']['scale'].detach(),constraint=torch.distributions.constraints.positive)
        with pyro.plate("intercept_plate",size=self.numObsTraits,dim=-1):
            pyro.sample("intercepts",dist.Normal(self.posteriorParamDict['intercepts']['mean'] ,self.posteriorParamDict['intercepts']['scale']))
        
        self.posteriorParamDict['latentPhenotypeEffectsPrecision']['conc']=pyro.param('latentPhenotypeEffectsPrecisionPosteriorConc',init_tensor=self.posteriorParamDict['latentPhenotypeEffectsPrecision']['conc'].detach(),constraint=torch.distributions.constraints.positive)
        self.posteriorParamDict['latentPhenotypeEffectsPrecision']['rate']=pyro.param('latentPhenotypeEffectsPrecisionPosteriorRate',init_tensor=self.posteriorParamDict['latentPhenotypeEffectsPrecision']['rate'].detach(),constraint=torch.distributions.constraints.positive)
        pyro.sample("latentPhenotypeEffectsPrecision",dist.Gamma(self.posteriorParamDict['latentPhenotypeEffectsPrecision']['conc'],self.posteriorParamDict['latentPhenotypeEffectsPrecision']['rate']))

        if self.mappingFunction=='Linear_Monotonic':
            self.posteriorParamDict['latentPhenotypeEffects']['conc']=pyro.param('latentPhenotypeEffectsPosteriorConc',init_tensor=self.posteriorParamDict['latentPhenotypeEffects']['conc'].detach(),constraint=torch.distributions.constraints.positive)
            self.posteriorParamDict['latentPhenotypeEffects']['rate']=pyro.param('latentPhenotypeEffectsPosteriorRates',init_tensor=self.posteriorParamDict['latentPhenotypeEffects']['rate'].detach(),constraint=torch.distributions.constraints.positive)
            pyro.sample("latentPhenotypeEffects",dist.Gamma(self.posteriorParamDict['latentPhenotypeEffects']['conc'],self.posteriorParamDict['latentPhenotypeEffects']['rate']).expand([1,self.numObsTraits]).to_event(1))
        else:
            self.posteriorParamDict['latentPhenotypeEffects']['mean']=pyro.param('latentPhenotypeEffectsPosteriorConc',init_tensor=self.posteriorParamDict['latentPhenotypeEffects']['mean'].detach())
            self.posteriorParamDict['latentPhenotypeEffects']['scale']=pyro.param('latentPhenotypeEffectsPosteriorScale',init_tensor=self.posteriorParamDict['latentPhenotypeEffects']['scale'].detach(),constraint=torch.distributions.constraints.positive)
            pyro.sample("latentPhenotypeEffects",dist.Normal(self.posteriorParamDict['latentPhenotypeEffects']['mean'],self.posteriorParamDict['latentPhenotypeEffects']['scale']).expand([1,self.numObsTraits]).to_event(1))

        self.posteriorParamDict['latentPhenotypePrevalence']['alpha']=pyro.param('latentPhenotypePrevalencePosteriorAlpha',init_tensor=self.posteriorParamDict['latentPhenotypePrevalence']['alpha'].detach(),constraint=torch.distributions.constraints.positive)
        self.posteriorParamDict['latentPhenotypePrevalence']['beta']=pyro.param('latentPhenotypePrevalencePosteriorBeta',init_tensor=self.posteriorParamDict['latentPhenotypePrevalence']['beta'].detach(),constraint=torch.distributions.constraints.positive)
        pyro.sample("latentPhenotypePrevalence",dist.Beta(self.posteriorParamDict['latentPhenotypePrevalence']['alpha'],self.posteriorParamDict['latentPhenotypePrevalence']['beta']))

        if self.numCovParam>0:
            self.posteriorParamDict['covEffects']['mean']=pyro.param("covEffectsPosteriorMean",init_tensor=self.posteriorParamDict['covEffects']['mean'].detach())
            self.posteriorParamDict['covEffects']['scale']=pyro.param("covEffectsPosteriorScale",init_tensor=self.posteriorParamDict['covEffects']['scale'].detach(),constraint=torch.distributions.constraints.positive)
            with pyro.plate("cov_plate",size=self.numCovParam):
                pyro.sample("covEffects",dist.Normal(self.posteriorParamDict['covEffects']['mean'],self.posteriorParamDict['covEffects']['scale']).to_event(1))


        if self.useAnchorDx > 0:

            self.posteriorParamDict['anchorDxNoise']['conc']=pyro.param('anchorDxNoisePosteriorConc',init_tensor=self.posteriorParamDict['anchorDxNoise']['conc'].detach(),constraint=torch.distributions.constraints.positive)
            self.posteriorParamDict['anchorDxNoise']['rate']=pyro.param('anchorDxNoisePosteriorRate',init_tensor=self.posteriorParamDict['anchorDxNoise']['rate'].detach(),constraint=torch.distributions.constraints.positive)
            pyro.sample('anchorDxNoise',dist.Gamma(self.posteriorParamDict['anchorDxNoise']['conc'],self.posteriorParamDict['anchorDxNoise']['rate']))

        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                latent_pheno_post = self.encoder(obs_data,*cat_cov_list)
                pyro.sample("latentPhenotypes", dist.Categorical(probs=latent_pheno_post))

    def PackageCurrentState(self,):
        packaged_model_state={}
        packaged_model_state['model_state'] = copy.deepcopy(self.state_dict())
        packaged_model_state['posterior_params'] = self._copyModelPosteriorParams()
        packaged_model_state['prior_params'] = self._copyModelPriorParams()
        return packaged_model_state

    def LoadPriorState(self,prior_model_state):
        self.load_state_dict(prior_model_state['model_state'],strict=True)
        self.posteriorParamDict=prior_model_state['posterior_params']
        self.priorParamDict=prior_model_state['prior_params']

    def _computeELBOPerDatum(self,obs_dis_array,cat_cov_list,anchor_dx_array,num_particles):
        elboFunc = TraceEnum_ELBO(num_particles=num_particles)
        elboVec = torch.zeros(anchor_dx_array.shape[0],dtype=torch.float32,device=self.compute_device)

        for model_trace, guide_trace in elboFunc._get_traces(self.model, self.guide,obs_dis_array,cat_cov_list,anchor_dx_array):
            elboVec+=torch.logsumexp(model_trace.nodes['obsTraitIncidence']['log_prob'].detach(),dim=0)/num_particles
            elboVec+=torch.logsumexp(model_trace.nodes['latentPhenotypes']['log_prob'].detach(),dim=0)/num_particles
            elboVec-=torch.logsumexp(guide_trace.nodes['latentPhenotypes']['log_prob'].detach(),dim=0)/num_particles
        return elboVec.reshape(elboVec.shape[0],1)

    def PredictLatentPhenotypes(self,obs_dis_array,cat_cov_list,anchor_dx_array=None,returnScale=False,num_particles=5,anchor_dx_prior = 0.5):
        if self.useAnchorDx:
            if anchor_dx_array is not None:
                output = self.encoder(torch.cat((obs_dis_array,anchor_dx_array),dim=-1),*cat_cov_list)
                exp_state=output
            else:
                if not torch.is_tensor(anchor_dx_prior):
                    anchor_dx_prior=torch.tensor(anchor_dx_prior,dtype=torch.float32,device =obs_dis_array.device)
                assert anchor_dx_prior.item()<1.0 and anchor_dx_prior.item()>0.0,"anchor dx prior must be between 0 and 1"
                elboWDx = self._computeELBOPerDatum(obs_dis_array,cat_cov_list,torch.ones((obs_dis_array.shape[0],1)),num_particles)
                elboWoDx = self._computeELBOPerDatum(obs_dis_array,cat_cov_list,torch.zeros((obs_dis_array.shape[0],1)),num_particles)
                probDx = torch.exp(elboWDx+torch.log(anchor_dx_prior)-torch.logsumexp(torch.cat((elboWDx+torch.log(anchor_dx_prior),elboWoDx+torch.log(1.0-anchor_dx_prior)),dim=1),dim=1,keepdim=True))

                output_on = self.encoder(torch.cat((obs_dis_array,torch.ones((obs_dis_array.shape[0],1))),dim=-1),*cat_cov_list)
                output_off = self.encoder(torch.cat((obs_dis_array,torch.zeros((obs_dis_array.shape[0],1))),dim=-1),*cat_cov_list)

                exp_state = probDx*output_on+(1.0-probDx)*output_off

        else:
            exp_state = self.encoder(obs_dis_array,*cat_cov_list)

        return exp_state
    
    def PredictAnchorDx(self,obs_dis_array,cat_cov_list,num_particles=5,anchor_dx_prior = 0.5):
        assert self.useAnchorDx, "Cannot predict anchor dx if model not trained using this information"
        if not torch.is_tensor(anchor_dx_prior):
            anchor_dx_prior=torch.tensor(anchor_dx_prior,dtype=torch.float32,device =obs_dis_array.device)
        assert anchor_dx_prior.item()<1.0 and anchor_dx_prior.item()>0.0,"anchor dx prior must be between 0 and 1"
        elboWDx = self._computeELBOPerDatum(obs_dis_array,cat_cov_list,torch.ones((obs_dis_array.shape[0],1)),num_particles)
        elboWoDx = self._computeELBOPerDatum(obs_dis_array,cat_cov_list,torch.zeros((obs_dis_array.shape[0],1)),num_particles)
        probDx = torch.exp(elboWDx+torch.log(anchor_dx_prior)-torch.logsumexp(torch.cat((elboWDx+torch.log(anchor_dx_prior),elboWoDx+torch.log(1.0-anchor_dx_prior)),dim=1),dim=1,keepdim=True))
        return probDx
        


