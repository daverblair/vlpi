#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:11:44 2020

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


class TranslationEncoder(nn.Module):
    
    def SwitchDevice(self,new_compute_device):
        self.compute_device=new_compute_device
        self.to(self.compute_device)
        
    def __init__(self,inputDecoder,nInputTraits:int,nOutputTraits:int, nCatList:Iterable[int],nLatentDim:int,**kwargs):
        super(TranslationEncoder,self).__init__()
        self.nInputTraits=nInputTraits
        self.nOutputTraits=nOutputTraits
        self.nCatList=nCatList
        self.nLatentDim = nLatentDim
        self.decoder=inputDecoder
        
        allKeywordArgs = list(kwargs.keys())
        
        if 'computeDevice' not in allKeywordArgs:
            """
            Specifies compute device for model fitting, can also be specified later by
            calling SwitchDevice

            """
            self.compute_device=None
        else:
            self.compute_device=kwargs['computeDevice']
            
            
        if 'encoderNetworkHyperparameters' not in allKeywordArgs:
            self.encoderHyperparameters={'n_layers' : 2, 'n_hidden' : 64, 'dropout_rate': 0.0, 'use_batch_norm':True}

        else:
            self.encoderHyperparameters = kwargs['encoderNetworkHyperparameters']
            assert isinstance(self.encoderHyperparameters,dict),"Expects dictionary of encoder hyperparameters"
            assert set(self.encoderHyperparameters.keys())==set(['n_layers','n_hidden','dropout_rate','use_batch_norm']),"Encoder hyperparameters must include: 'n_layers','n_hidden','dropout_rate','use_batch_norm'"
            
            
        if self.dropLinearCovariateColumn:
            self.numCovParam = sum(self.nCatList)-len(self.nCatList)
        else:
            self.numCovParam = sum(self.nCatList)
            
        self.encoder=MeanScaleEncoder(self.nInputTraits, self.nLatentDim, n_cat_list=self.numInputCatList, **self.encoderHyperparameters)
        
        if self.compute_device is not None:
            self.SwitchDevice(self.compute_device)
        self.eval()
        
    def model(self, decoded_data=None, cat_cov_list=None,label_dx=None,encoded_data=None,numSamples=None,minibatch_scale=1.0, annealing_factor=1.0):
        if decoded_data is not None:
            numSamples=decoded_data.shape[0]
        else:
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
                pyro.sample("obsTraitIncidence",dist.Bernoulli(latent_dx_prob).to_event(1),obs=decoded_data)
                
                
    def guide(self,decoded_data=None, cat_cov_list=None,label_dx=None,encoded_data=None,numSamples = None,minibatch_scale=1.0,annealing_factor=1.0):
        if encoded_data is not None:
            numSamples=encoded_data.shape[0]
        else:
            print("Warning: Passing TranslationEncoder an empty set of encodings. Using decoder instead. This should only be done for debugging purposes.")
            encoded_data=decoded_data
        assert encoded_data.shape[0]==decoded_data.shape[0],"Encoding and decoding data must have the same number of samples."

        pyro.module("encoder", self.encoder,update_module_params=False)


        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                z_mean,z_std = self.encoder(encoded_data,*cat_cov_list)
                with pyro.poutine.scale(None, annealing_factor):
                    pyro.sample("latentPhenotypes", dist.Normal(z_mean, z_std).to_event(1))

            
        