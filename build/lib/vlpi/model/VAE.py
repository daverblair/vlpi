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
        """

        Switches model to a different device.

        Parameters
        ----------
        new_compute_device : int or str
            Parameter that specifies the device. Can be integer (GPU) or str ('cpu')

        """
        self.compute_device=new_compute_device
        self.to(self.compute_device)


    def __init__(self,numObsTraits:int, numCatList:Iterable[int],nLatentDim:int,decoderType:str,**kwargs):
        """

        Variational autoencoder used for latent phenotype inference.

        Parameters
        ----------
        numObsTraits : int
            Number of traits or symptoms used in the model.
        numCatList : Iterable[int]
            List containing the number of categories for each categorical covariate.
        nLatentDim : int
            Number of latent dimensions in the model
        decoderType : str
            Type of decoder. Must be one of the following: 'Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'
        **kwargs : type
            Mutliple kwargs available. Please see source code for details.

        Returns
        -------
        Nonlinear

        """
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
            self.decoderHyperparameters={'n_layers' : 2, 'n_hidden' : 64, 'dropout_rate': 0.0, 'use_batch_norm':True}

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


    def model(self, obs_data=None, cat_cov_list=None,label_dx=None,encoded_data=None,numSamples=None,minibatch_scale=1.0, annealing_factor=1.0):
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


    def guide(self,obs_data=None, cat_cov_list=None,label_dx=None,encoded_data=None,numSamples = None,minibatch_scale=1.0,annealing_factor=1.0):
        if obs_data is not None:
            numSamples=obs_data.shape[0]
        else:
            pass

        pyro.module("encoder", self.encoder,update_module_params=False)


        with pyro.poutine.scale(None,minibatch_scale):
            with pyro.plate("latent_pheno_plate",size=numSamples):
                z_mean,z_std = self.encoder(obs_data,*cat_cov_list)
                with pyro.poutine.scale(None, annealing_factor):
                    pyro.sample("latentPhenotypes", dist.Normal(z_mean, z_std).to_event(1))



    def ComputeELBOPerDatum(self,obs_dis_array,cat_cov_list,num_particles=10):
        """

        Computes the evidence lower bound (ELBO) for each observation in the dataset.

        Parameters
        ----------
        obs_dis_array : torch.tensor
            Binary array of observed data.
        cat_cov_list : list of torch.tensor
            List of categorical covariate values for the dataset
        num_particles : int
            Number of particles (samples) used to approximate the ELBOs

        Returns
        -------
        torch.tensor
            Per-datum ELBOs

        """
        elboFunc = Trace_ELBO(num_particles=num_particles)
        elboVec = torch.zeros(obs_dis_array.shape[0],dtype=torch.float32,device=self.compute_device)

        for model_trace, guide_trace in elboFunc._get_traces(self.model, self.guide,(obs_dis_array,cat_cov_list),{}):
            elboVec+=model_trace.nodes['obsTraitIncidence']['log_prob'].detach()/num_particles
            elboVec+=model_trace.nodes['latentPhenotypes']['log_prob'].detach()/num_particles
            elboVec-=guide_trace.nodes['latentPhenotypes']['log_prob'].detach()/num_particles
        return elboVec.reshape(elboVec.shape[0],1)

    def PredictLatentPhenotypes(self,obs_dis_array,cat_cov_list,returnScale=False,num_particles=10):
        """
        Produces estimates of posterior mean (and scale) of latent phenotypes given some set of observations.

        Parameters
        ----------
        obs_dis_array : torch.tensor
            Binary array of observed data.
        cat_cov_list : list of torch.tensor
            List of categorical covariate values for the dataset
        returnScale : bool
            Whether or not to include scale of posterior as well.
        num_particles : int
            Number of particles (samples) used to approximate the ELBOs

        Returns
        -------
        torch.tensor
            2-D (or 3-D if including scale) array of posterior distribution means (plus scale)

        """

        z_mean,z_scale = self.encoder(obs_dis_array,*cat_cov_list)
        if returnScale:
            return z_mean,z_scale
        else:
            return z_mean


    def PackageCurrentState(self):
        """
        Packages the model state dict as a dictionary.

        Returns
        -------
        dict
            Model state dict.

        """
        packaged_model_state={}
        packaged_model_state['model_state'] = copy.deepcopy(self.state_dict(keep_vars=True))
        return packaged_model_state

    def LoadPriorState(self,prior_model_state):
        """
        Loads model state from dictionary

        Parameters
        ----------
        prior_model_state : dict
            Dictionary of model state produced by PackageCurrentState

        Returns
        -------
        None

        """
        self.load_state_dict(prior_model_state['model_state'],strict=True)
