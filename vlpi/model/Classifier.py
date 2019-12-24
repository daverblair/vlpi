#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:51:54 2019

@author: davidblair
"""
import torch
import pyro
import torch.nn as nn
import pyro.distributions as dist
from typing import Iterable
from vlpi.model.Decoders import  LinearDecoder_Monotonic,LinearDecoder,NonlinearMLPDecoder,NonlinearMLPDecoder_Monotonic
from vlpi.utils.UtilityFunctions import random_catcov
import copy

class Classifier(nn.Module):

    def SwitchDevice(self,new_compute_device):
        self.compute_device=new_compute_device
        self.to(self.compute_device)
        

    def __init__(self, numObsTraits:int, numCatList:Iterable[int],decoderType:str,**kwargs):


        """
        DisciminativeModel uses known cases to learn a mapping function from observed traits to some binary phenotype of interest
        through a probit link function. In the linear setting, it is equivalent to probit regression.
        
        numObsTraits-->number of phenotypes included into the model
        numCatList-->List of the number of categories for each categorical covariate; total # of categorical covariates = len(numCatList).
        decoderType-->type of function mapping observed to latent phenotype space. Options include: Linear, Linear_Monotonic, Nonlinear, Nonlinear_Monotonic.
        
        **kwargs: There are multiple keyword arguments available as well. These handle many of the hyperparameters and more esoteric modeling options. Information regarding each is provided below
        """

        super(Classifier,self).__init__()
        self.numObsTraits=numObsTraits
        self.numCatList=numCatList
        self.decoderType=decoderType

        assert self.decoderType in ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'], "Only Linear, Linear_Monotonic, Nonlinear, and Nonlinear_Monotonic models are supported by the DiscriminativeModel"
        #kwargs
        
        allKeywordArgs = list(kwargs.keys())
        
        
        if 'linkFunction' not in allKeywordArgs:
            self.linkFunction = lambda x:torch.sigmoid(x)
        else:
            linkFunction = kwargs['linkFunction']
            assert linkFunction in ['Logit','Probit'],"Only Logit and Probit link functions currently supported."
            
            if linkFunction=='Logit':
                self.linkFunction = lambda x:torch.sigmoid(x)
            else:
                self.linkFunction = lambda x:dist.Normal(0.0,1.0).cdf(x)
        
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
        

            
        if 'decoderNetworkHyperparameters' not in allKeywordArgs:
            self.decoderHyperparameters={'n_layers' : 2, 'n_hidden' : 32, 'dropout_rate': 0.1, 'use_batch_norm':True}

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
                    
        if self.decoderType=='Linear':
            self.decoder = LinearDecoder(self.numObsTraits,self.numCatList,1,self.dropLinearCovariateColumn,True)
        elif self.decoderType=='Linear_Monotonic':
            self.decoder = LinearDecoder_Monotonic(self.numObsTraits,self.numCatList,1,self.dropLinearCovariateColumn,True)
        elif self.decoderType == 'Nonlinear':
            self.decoder = NonlinearMLPDecoder(self.numObsTraits,self.numCatList,1,self.dropLinearCovariateColumn,self.coupleCovariates,**self.decoderHyperparameters)
        else:
            self.decoder = NonlinearMLPDecoder_Monotonic(self.numObsTraits,self.numCatList,1,self.dropLinearCovariateColumn,**self.decoderHyperparameters)
        
        if self.compute_device is not None:
            self.SwitchDevice(self.compute_device)
        self.eval()

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


    def PredictLabels(self,dis_array,covariate_list):
        """
        Returns expected value for probability of having anchor disease
        """

        return  self.linkFunction(self.decoder.forward(dis_array,*covariate_list))


    def PackageCurrentState(self):
        packaged_model_state={}
        packaged_model_state['model_state'] = copy.deepcopy(self.state_dict(keep_vars=True))
        return packaged_model_state

    def LoadPriorState(self,prior_model_state):
        self.load_state_dict(prior_model_state['model_state'],strict=True)
        
