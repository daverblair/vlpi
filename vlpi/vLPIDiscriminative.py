#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:51:54 2019

@author: davidblair
"""
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from typing import Iterable
from vlpi.Decoders import LinearDecoder,LinearDecoder_Monotonic,NonlinearMLPDecoder,NonlinearMLPDecoder_Monotonic
from vlpi.utils import random_catcov,infer_beta_from_CI
import copy

class DiscriminativeModel(nn.Module):

    def SwitchDevice(self,new_compute_device):
        self.compute_device=new_compute_device
        self.to(self.compute_device)
        
    def _returnUpdatingParamsFlattened(self,withPosterior=True,withEncoder=True):
        all_params=[]
        for p_vals in self.decoder.parameters():
            all_params+=[p_vals.detach().flatten()]
        return torch.cat(all_params)

    def __init__(self, numObsTraits:int, numCatList:Iterable[int],mappingFunction:str,linkFunction:str,**kwargs):


        """
        DisciminativeModel uses known cases to learn a mapping function from observed traits to some binary phenotype of interest
        through a probit link function. In the linear setting, it is equivalent to probit regression.
        
        numObsTraits-->number of phenotypes included into the model
        numCatList-->List of the number of categories for each categorical covariate; total # of categorical covariates = len(numCatList).
        mappingFunction-->type of function mapping observed to latent phenotype space. Options include: Linear, Linear_Monotonic, Nonlinear, Nonlinear_Monotonic.
        linkFunction-->function used to link continuous prediction scores to expectation of binary RV. Choices are ['Logit','Probit']
        
        **kwargs: There are multiple keyword arguments available as well. These handle many of the hyperparameters and more esoteric modeling options. Information regarding each is provided below
        """

        super(DiscriminativeModel,self).__init__()
        self.numObsTraits=numObsTraits
        self.numCatList=numCatList
        self.mappingFunction=mappingFunction

        assert self.mappingFunction in ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'], "Only Linear, Linear_Monotonic, Nonlinear, and Nonlinear_Monotonic models are supported by the DiscriminativeModel"
        #kwargs
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

        if 'coupleCovariates' not in allKeywordArgs:
            """
            Specifies whether to couple covariate variables into Nonlinear NN mapping function. Defaults to True for
            Discriminative model
            """
            self.coupleCovariates=False
        else:
            self.coupleCovariates=kwargs['coupleCovariates']
            assert isinstance(self.coupleCovariates,bool),"coupleCovariates expects boolean value"
            
        if 'neuralNetworkHyperparameters' not in allKeywordArgs:
            self.decoderHyperparameters={'n_layers' : 2, 'n_hidden' : 64, 'dropout_rate': 0.5, 'use_batch_norm':True}
        else:
            self.decoderHyperparameters=kwargs['neuralNetworkHyperparameters']
            assert isinstance(self.decoderHyperparameters,dict),"Expects dictionary of decoder hyperparameters"
            assert set(self.decoderHyperparameters.keys())==set(['n_layers','n_hidden','dropout_rate','use_batch_norm']),"Encoder hyperparameters must include: 'n_layers','n_hidden','dropout_rate','use_batch_norm'"



        if self.mappingFunction=='Linear':
            self.decoder = LinearDecoder(self.numObsTraits,self.numCatList,1,self.dropLinearCovariateColumn,True)
        elif self.mappingFunction=='Linear_Monotonic':
            self.decoder=LinearDecoder_Monotonic(self.numObsTraits,self.numCatList,1,self.dropLinearCovariateColumn,True)
        elif self.mappingFunction=='Nonlinear':
            self.decoder=NonlinearMLPDecoder(self.numObsTraits,self.numCatList,1,self.dropLinearCovariateColumn,self.coupleCovariates,**self.decoderHyperparameters)
        elif self.mappingFunction=='Nonlinear_Monotonic':
            self.decoder=NonlinearMLPDecoder_Monotonic(self.numObsTraits,self.numCatList,1,self.dropLinearCovariateColumn,**self.decoderHyperparameters)

        if self.dropLinearCovariateColumn:
            self.numCovParam = sum(self.numCatList)-len(self.numCatList)
        else:
            self.numCovParam = sum(self.numCatList)
        
        self.linkFunctionType = linkFunction
        if self.linkFunctionType=='Probit':
            self.linkFunction = lambda x:dist.Normal(0.0,1.0).cdf(x)
        else:
            self.linkFunction = lambda x:torch.sigmoid(x)
            
        self.to(self.compute_device)
        self.eval()

    def SimulateData(self,numSamples,obsDisPrevalencePrior95_CI = [0.1,0.25]):

        """
        Draws numSamples of samples from model.
        numSamples: number of samples
        obsDisPrevalencePrior95_CI: 95%CI for the generating distribution from which to draw the observed disease prevalence rates.
        """

        cat_cov_list = []
        for n_cat in self.numCatList:
            cat_cov_list+=[random_catcov(n_cat,numSamples,self.compute_device)]

        betaPrevPrior=torch.tensor(infer_beta_from_CI(obsDisPrevalencePrior95_CI),dtype=torch.float32,device=self.compute_device)
        prevRates = dist.Beta(betaPrevPrior[0]*torch.ones(1,self.numObsTraits,dtype=torch.float32,device=self.compute_device),betaPrevPrior[1]*torch.ones(1,self.numObsTraits,dtype=torch.float32,device=self.compute_device)).sample()
        obs_dis_data = dist.Bernoulli(torch.ones(numSamples,1,dtype=torch.float32,device=self.compute_device)*prevRates).sample()


        dis_liability = self.decoder(obs_dis_data,*cat_cov_list)
        anchor_dx_prob =  self.linkFunction(dis_liability)
        anchor_dx=dist.Bernoulli(anchor_dx_prob).sample()
        
        outputDict={}
        outputDict['incidence_data']=obs_dis_data
        if self.numCovParam>0:
            outputDict['covariate_data']=cat_cov_list
        else:
            outputDict['covariate_data']=[]
        outputDict['anchor_dx_data'] = anchor_dx
    
        return outputDict

    def model(self,obs_data=None, cat_cov_list=None,anchor_dx=None,sample_scores=None,minibatch_scale=1.0):
        assert obs_data is not None,"Model requires input data. To simulate random samples, use SimulateData function."
        assert anchor_dx is not None,"Model requires input of latent dx data"
        assert anchor_dx.shape[0]==obs_data.shape[0],"Dimensions of anchor and observed dx data must match."
        numSamples = obs_data.shape[0]
        #load the decoder model into the pyro parameter space, enables updating
        pyro.module("decoder", self.decoder)
        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("anchor_dx_plate",size=numSamples):
                liabilities = self.decoder.forward(obs_data,*cat_cov_list)
                latent_dx_prob = self.linkFunction(liabilities)
                pyro.sample("anchorDisDx",dist.Bernoulli(latent_dx_prob).to_event(1),obs=anchor_dx)

    def guide(self,obs_data=None, cat_cov_list=None,anchor_dx=None,sample_scores=None,minibatch_scale=1.0):
        assert obs_data is not None,"Model requires input data. To simulate random samples, use SimulateData function."
        assert anchor_dx is not None,"Model requires input of latent dx data"
        assert anchor_dx.shape[0]==obs_data.shape[0],"Dimensions of anchor and observed dx data must match."


    def PredictAnchorDx(self,dis_array,covariate_list):
        """
        Returns expected value for probability of having anchor disease
        """

        return  self.linkFunction(self.decoder.forward(dis_array,*covariate_list))


    def PackageCurrentState(self):
        packaged_model_state={}
        packaged_model_state['model_state'] = copy.deepcopy(self.state_dict())
        packaged_model_state['posterior_params'] = None
        packaged_model_state['prior_params'] = None
        return packaged_model_state

    def LoadPriorState(self,prior_model_state):
        self.load_state_dict(prior_model_state['model_state'],strict=True)
        
