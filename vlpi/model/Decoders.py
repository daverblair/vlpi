#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:26:37 2019

@author: davidblair
"""
import torch
import torch.nn as nn
from typing import Iterable
from vlpi.model.FCLayers import FCLayers,FCLayers_Monotonic
from vlpi.utils.LinearPositiveWeights import LinearPositiveWeights
from vlpi.utils.UtilityFunctions import build_onehot_arrays,random_catcov


class LinearDecoder_Monotonic(nn.Module):


    def __init__(self,nInputDim:int,nCatCovList:Iterable[int],nOutputDim:int,dropLinearCovariateColumn:bool,includeBias:bool):
        """
        Monotonic linear (stritly positive weights) decoder network mapping nInputDim +len(nCatCovList)
        inputs to nOutputDim nodes.

        dropLinearCovariateColumn: whether to drop one value from each categorical covariate
        includeBias: whether or not to include a bias term (intercept) into the network

        """

        super(LinearDecoder_Monotonic, self).__init__()
        self.nCatCovList=nCatCovList
        self.nOutputDim=nOutputDim
        self.dropLinearCovariateColumn=dropLinearCovariateColumn
        self.nInputDim=nInputDim
        if len(nCatCovList)>0:
            if dropLinearCovariateColumn:
                self.nInputDimCov=sum(nCatCovList)-len(nCatCovList)
            else:
                self.nInputDimCov=sum(nCatCovList)
            self.linear_cov = nn.Linear(self.nInputDimCov,nOutputDim,bias=False)
        self.linear_latent = LinearPositiveWeights(self.nInputDim,nOutputDim,bias=includeBias)



    def forward(self,latent_var,*cat_cov_list:int):

        one_hot_cat_list = build_onehot_arrays(cat_cov_list,self.nCatCovList,self.dropLinearCovariateColumn)
        latent_component = self.linear_latent(latent_var)
        if len(one_hot_cat_list)>0:
            return self.linear_cov(torch.cat((*one_hot_cat_list,),dim=-1))+latent_component
        else:
            return latent_component


class LinearDecoder(nn.Module):
    def __init__(self,nInputDim:int,nCatCovList:Iterable[int],nOutputDim:int,dropLinearCovariateColumn:bool,includeBias:bool):
        """
        Linear decoder network mapping nInputDim +len(nCatCovList) inputs to
        nOutputDim nodes.

        dropLinearCovariateColumn: whether to drop one value from each categorical covariate
        includeBias: whether or not to include a bias term (intercept) into the network

        """
        super(LinearDecoder, self).__init__()
        self.nCatCovList=nCatCovList
        self.nOutputDim=nOutputDim
        self.dropLinearCovariateColumn=dropLinearCovariateColumn
        self.nInputDim=nInputDim
        if len(nCatCovList)>0:
            if dropLinearCovariateColumn:
                self.nInputDimCov=sum(nCatCovList)-len(nCatCovList)
            else:
                self.nInputDimCov=sum(nCatCovList)
            self.linear_cov = nn.Linear(self.nInputDimCov,nOutputDim,bias=False)
        self.linear_latent = nn.Linear(self.nInputDim,nOutputDim,bias=includeBias)


    def forward(self,latent_var,*cat_cov_list:int):
        one_hot_cat_list = build_onehot_arrays(cat_cov_list,self.nCatCovList,self.dropLinearCovariateColumn)
        if len(one_hot_cat_list)>0:
            return self.linear_cov(torch.cat((*one_hot_cat_list,),dim=-1))+self.linear_latent(latent_var)
        else:
            return self.linear_latent(latent_var)

class NonlinearMLPDecoder(nn.Module):
    def __init__(self,nInputDim:int,nCatCovList:Iterable[int],nOutputDim:int,dropLinearCovariateColumn:bool,coupleCovariates:bool,n_layers:int=2,n_hidden:int=128,dropout_rate:float=0.2,use_batch_norm:bool=True):
        """
        Nonlinear MLP decoder network mapping nInputDim +len(nCatCovList) inputs to
        nOutputDim nodes.
        """
        super(NonlinearMLPDecoder, self).__init__()
        self.nCatCovList=nCatCovList
        self.nOutputDim=nOutputDim
        self.dropLinearCovariateColumn=dropLinearCovariateColumn
        self.nInputDim=nInputDim
        self.coupleCovariates = coupleCovariates


        if self.coupleCovariates==True:
            self.nonlinear = FCLayers(n_in=self.nInputDim,n_out=n_hidden,n_cat_list=nCatCovList,n_layers=n_layers,n_hidden=n_hidden,dropout_rate=dropout_rate,use_batch_norm=use_batch_norm)
            self.output_layer =  nn.Linear(n_hidden, self.nOutputDim, bias = True)
        else:
            self.nonlinear_latent = FCLayers(n_in=self.nInputDim,n_out=n_hidden,n_cat_list=[],n_layers=n_layers,n_hidden=n_hidden,dropout_rate=dropout_rate,use_batch_norm=use_batch_norm)
            self.output_layer =  nn.Linear(n_hidden, self.nOutputDim,bias=True)
            if len(nCatCovList)>0:
                if dropLinearCovariateColumn:
                    self.nInputDimCov=sum(nCatCovList)-len(nCatCovList)
                else:
                    self.nInputDimCov=sum(nCatCovList)
                self.linear_cov = nn.Linear(self.nInputDimCov,nOutputDim,bias=False)


    def forward(self,latent_var,*cat_cov_list:Iterable[int]):
        if self.coupleCovariates==True:
            return self.output_layer(self.nonlinear(latent_var,*cat_cov_list))
        else:
            one_hot_cat_list = build_onehot_arrays(cat_cov_list,self.nCatCovList,self.dropLinearCovariateColumn)
            if len(one_hot_cat_list)>0:
                return self.output_layer(self.nonlinear_latent(latent_var,[]))+self.linear_cov(torch.cat((*one_hot_cat_list,),dim=-1))
            else:
                return self.output_layer(self.nonlinear_latent(latent_var,[]))


class NonlinearMLPDecoder_Monotonic(nn.Module):
    def __init__(self,nInputDim:int,nCatCovList:Iterable[int],nOutputDim:int,dropLinearCovariateColumn:bool,n_layers:int=2,n_hidden:int=128,dropout_rate:float=0.2,use_batch_norm:bool=False):
        """
        Nonlinear, monotonic decoder network mapping nInputDim +len(nCatCovList) inputs to
        nOutputDim nodes.

        dropLinearCovariateColumn: whether to drop one value from each categorical covariate
        includeBias: whether or not to include a bias term (intercept) into the network

        """

        super(NonlinearMLPDecoder_Monotonic, self).__init__()
        self.nCatCovList=nCatCovList
        self.nOutputDim=nOutputDim
        self.dropLinearCovariateColumn=dropLinearCovariateColumn
        self.nInputDim=nInputDim

        if len(nCatCovList)>0:
            if dropLinearCovariateColumn:
                self.nInputDimCov=sum(nCatCovList)-len(nCatCovList)
            else:
                self.nInputDimCov=sum(nCatCovList)
            self.linear_cov = nn.Linear(self.nInputDimCov,nOutputDim,bias=False)

        self.nonlinear_pos_latent = FCLayers_Monotonic(n_in=self.nInputDim,n_out=n_hidden,n_cat_list=[],n_layers=n_layers,n_hidden=n_hidden,dropout_rate=dropout_rate)
        self.output_layer =  LinearPositiveWeights(n_hidden, self.nOutputDim,bias=True)

    def forward(self,latent_var,*cat_cov_list:Iterable[int]):
        one_hot_cat_list = build_onehot_arrays(cat_cov_list,self.nCatCovList,self.dropLinearCovariateColumn)
        if len(one_hot_cat_list)>0:
            return self.output_layer(self.nonlinear_pos_latent(latent_var,[]))+self.linear_cov(torch.cat((*one_hot_cat_list,),dim=-1))
        else:
            return self.output_layer(self.nonlinear_pos_latent(latent_var,[]))
