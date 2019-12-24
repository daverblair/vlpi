#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:42:20 2019

@author: davidblair
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO
from typing import Iterable
import copy
from vlpi.model.Encoders import MeanScaleEncoder
from vlpi.model.Decoders import LinearDecoder_Monotonic,LinearDecoder,NonlinearMLPDecoder,NonlinearMLPDecoder_Monotonic
from vlpi.utils.UtilityFunctions import random_catcov


class VAE(nn.Module):

    
    def SwitchDevice(self,new_compute_device):
        self.compute_device=new_compute_device
        self.to(self.compute_device)
        
    def _encoder_only_model(self,obs_data=None, cat_cov_list=None,label_dx=None, sample_scores=None,numSamples=None,minibatch_scale=1.0,annealing_factor=1.0):
        
        
        assert sample_scores is not None, "Sample scores must be included in order to train the encoder individually."
        assert obs_data is not None, "obs_data must be included when using the _encoder_only_model."
        pyro.module("encoder", self.encoder,update_module_params=False)

        numSamples=obs_data.shape[0]

        latent_vals = torch.zeros(sample_scores.shape,dtype=torch.float32,device=sample_scores.device)
        norm_vals = torch.arange(sample_scores.shape[0],device=sample_scores.device).to(dtype=torch.float32)
        
        norm_vals = dist.Normal(torch.tensor(0.0,device=sample_scores.device,dtype=torch.float32),torch.tensor(1.0,device=sample_scores.device,dtype=torch.float32)).icdf(1.0-(norm_vals+1.0)/(norm_vals.shape[0]+1.0))
        for i in range(self.nLatentDim):
            rankings=sample_scores[:,i].argsort(dim=0,descending=True)
            latent_vals[rankings,i]=norm_vals

        #add noise to latent states to prevent overtraining the variance
        latent_vals+=dist.Normal(torch.tensor(0.0,dtype=torch.float32,device=sample_scores.device),torch.tensor(1.0,dtype=torch.float32,device =sample_scores.device)).sample(latent_vals.shape)

        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("rank_pheno_plate",size=numSamples):
                z_mean,z_std = self.encoder(obs_data,*cat_cov_list)
                pyro.sample("rankPhenotypes",dist.Normal(z_mean, z_std).to_event(1),obs=latent_vals)


    def _encoder_only_guide(self,obs_data=None, cat_cov_list=None,label_dx=None, sample_scores=None,numSamples=None,minibatch_scale=1.0,annealing_factor=1.0):
        assert sample_scores is not None, "Sample scores must be included in order to train the encoder individually."
        assert obs_data is not None, "obs_data must be included when using the _encoder_only_model."
        
    def __init__(self,numObsTraits:int, numCatList:Iterable[int],nLatentDim:int,decoderType:str,**kwargs):
        super(VAE,self).__init__()
        self.numObsTraits=numObsTraits
        self.nLatentDim = nLatentDim
        self.numCatList=numCatList
        self.decoderType=decoderType
        
        assert self.decoderType in ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'], "Currently supported decoders for VAE include: 'Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'"
        
        allKeywordArgs = list(kwargs.keys())
        
        
        if 'linkFunction' not in allKeywordArgs:
            self.linkFunction = lambda x:torch.sigmoid(x)
        else:
            linkFunction = kwargs['linkFunction']
            assert linkFunction in ['Logit','Probit'],"Only Logit and Probit link functions currently supported."
            
            if linkFunction=='Logit':
                self.linkFunction = lambda x:torch.sigmoid(x)
            else:
                self.linkFunction = lambda x:dist.Normal(torch.tensor(0.0,dtype=torch.float32,device=x.device),torch.tensor(1.0,dtype=torch.float32,device=x.device)).cdf(x)
        
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
            
        if 'coupleCovariates' not in allKeywordArgs:
            """
            Specifies whether to couple covariates to non-linear MLP network (True), or to model them  using an independent linear network (False). Defaults to True.
            """
            self.coupleCovariates=True
        else:
            self.coupleCovariates=kwargs['coupleCovariates']
            assert isinstance(self.dropLinearCovariateColumn,bool),"coupleCovariates expects boolean value"
            if self.decoderType not in ['Nonlinear']:
                print("Warning: Not fitting Nonlinear model. Coupling covariates has no effect on inference.")
        
            
        if 'encoderNetworkHyperparameters' not in allKeywordArgs:
            self.encoderHyperparameters={'n_layers' : 2, 'n_hidden' : 64, 'dropout_rate': 0.0, 'use_batch_norm':True}

        else:
            self.encoderHyperparameters = kwargs['encoderNetworkHyperparameters']
            assert isinstance(self.encoderHyperparameters,dict),"Expects dictionary of encoder hyperparameters"
            assert set(self.encoderHyperparameters.keys())==set(['n_layers','n_hidden','dropout_rate','use_batch_norm']),"Encoder hyperparameters must include: 'n_layers','n_hidden','dropout_rate','use_batch_norm'"
            
        if 'decoderNetworkHyperparameters' not in allKeywordArgs:
            self.decoderHyperparameters={'n_layers' : 2, 'n_hidden' : 32, 'dropout_rate': 0.0, 'use_batch_norm':True}

        else:
            self.decoderHyperparameters = kwargs['decoderNetworkHyperparameters']
            assert isinstance(self.encoderHyperparameters,dict),"Expects dictionary of decoder hyperparameters"
            assert set(self.encoderHyperparameters.keys())==set(['n_layers','n_hidden','dropout_rate','use_batch_norm']),"Decoder hyperparameters must include: 'n_layers','n_hidden','dropout_rate','use_batch_norm'"
            
            if self.decoderType not in ['Nonlinear','Nonlinear_Monotonic']:
                print("Warning: Decoder neural network hyperparameters specified for a linear model. Parameters will not be used.")
        
        if self.dropLinearCovariateColumn:
            self.numCovParam = sum(self.numCatList)-len(self.numCatList)
        else:
            self.numCovParam = sum(self.numCatList)
            
        self.encoder=MeanScaleEncoder(self.numObsTraits, self.nLatentDim, n_cat_list=self.numCatList, **self.encoderHyperparameters)
        
        if self.decoderType=='Linear':
            self.decoder = LinearDecoder(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn,True)
        elif self.decoderType=='Linear_Monotonic':
            self.decoder = LinearDecoder_Monotonic(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn,True)
        elif self.decoderType == 'Nonlinear':
            self.decoder = NonlinearMLPDecoder(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn,self.coupleCovariates,**self.decoderHyperparameters)
        else:
            self.decoder = NonlinearMLPDecoder_Monotonic(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn,**self.decoderHyperparameters)
        
        if self.compute_device is not None:
            self.SwitchDevice(self.compute_device)
        self.eval()
        
    def model(self, obs_data=None, cat_cov_list=None,label_dx=None,sample_scores=None,numSamples=None,minibatch_scale=1.0, annealing_factor=1.0):
        if obs_data is not None:
            numSamples=obs_data.shape[0]
        else:
                #assert numSamples is not None, "Must provide either observed data or number of samples to simulate."
            if numSamples is None:
                numSamples=1000
                print('Warning: no arguments were given to VAE.model. This should only be done during debugging.')

            cat_cov_list = []
            for n_cat in self.numCatList:
                cat_cov_list+=[random_catcov(n_cat,numSamples,device=self.compute_device)]
                
        pyro.module("decoder", self.decoder,update_module_params=False)
        
        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                with pyro.poutine.scale(None, annealing_factor):
                    latentPhenotypes=pyro.sample("latentPhenotypes",dist.Normal(torch.zeros(1,self.nLatentDim,dtype=torch.float32,device=self.compute_device),torch.ones(1,self.nLatentDim,dtype=torch.float32,device=self.compute_device)).to_event(1))
                
                liability_vals = self.decoder.forward(latentPhenotypes,*cat_cov_list)
                latent_dx_prob = self.linkFunction(liability_vals)
                pyro.sample("obsTraitIncidence",dist.Bernoulli(latent_dx_prob).to_event(1),obs=obs_data)


    def guide(self,obs_data=None, cat_cov_list=None,label_dx=None,sample_scores=None,numSamples = None,minibatch_scale=1.0,annealing_factor=1.0):
        if obs_data is not None:
            numSamples=obs_data.shape[0]

        else:
            pass

        if sample_scores is None:
            pyro.module("encoder", self.encoder,update_module_params=False)


        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                z_mean,z_std = self.encoder(obs_data,*cat_cov_list)
                with pyro.poutine.scale(None, annealing_factor):
                    pyro.sample("latentPhenotypes", dist.Normal(z_mean, z_std).to_event(1))
                
    

    def ComputeELBOPerDatum(self,obs_dis_array,cat_cov_list,num_particles=10):
        elboFunc = Trace_ELBO(num_particles=num_particles)
        elboVec = torch.zeros(obs_dis_array.shape[0],dtype=torch.float32,device=self.compute_device)

        for model_trace, guide_trace in elboFunc._get_traces(self.model, self.guide,(obs_dis_array,cat_cov_list),{}):
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
        return packaged_model_state

    def LoadPriorState(self,prior_model_state):
        self.load_state_dict(prior_model_state['model_state'],strict=True)


if __name__=='__main__':
    
    from vlpi.data.ClinicalDataSimulator import ClinicalDataSimulator
    from vlpi.data.ClinicalDataset import ClinicalDatasetSampler,ClinicalDataset 
    
    
    from pyro import poutine
    pyro.enable_validation(True)    # <---- This is always a good idea!

    # We'll ue this helper to check our models are correct.
    def test_model(model, guide, loss):
        pyro.clear_param_store()
        print(loss.loss(model, guide))
        
        
    numSamples = 100
    numAssociatedTraits=20
    nLatentSimDim=4
    nLatentFitDim=4
    mappingFunction='Linear_Monotonic'
    numCovPerClass = [2,3,10] 
    covNames = ['A','B','C']

    
    
    simulator = ClinicalDataSimulator(numAssociatedTraits,nLatentSimDim,numCatList=numCovPerClass)
    simData=simulator.GenerateClinicalData(numSamples,0.0)
    
    clinData = ClinicalDataset()
    
    disList = list(clinData.dxCodeToDataIndexMap.keys())[0:numAssociatedTraits]
    clinData.IncludeOnly(disList)
    
        
        
        
    clinData.LoadFromArrays(simData['incidence_data'],simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')
    sampler = ClinicalDatasetSampler(clinData,0.75,returnArrays='Torch')
    
    vae=VAE(numAssociatedTraits, numCovPerClass,nLatentSimDim,mappingFunction)
    
    trace = poutine.trace(vae.model).get_trace()
    
    trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    print(trace.format_shapes())
    
    def model():
        return vae.model(*sampler.ReturnFullTrainingDataset(randomize=False))
    
    def guide():
        return vae.guide(*sampler.ReturnFullTrainingDataset(randomize=False))
    
    test_model(model,guide,loss=Trace_ELBO())