#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:26:37 2019

@author: davidblair
"""
import torch
import torch.nn as nn

from FCLayers import FCLayers,FCLayers_Monotonic
from LinearPositiveWeights import LinearPositiveWeights
from utils import build_onehot_arrays,random_catcov
from typing import Iterable



class LinearDecoder_Monotonic(nn.Module):

    
    def __init__(self,nLatentDim:int,nCatCovList:Iterable[int],nOutputDim:int,dropLinearCovariateColumn:bool):
        """
        Creates a single layer, linear neural network mapping nLatentDim latent variables and sum(nCatCovList) categorical 
        covariates to an output space of nOutputDim, under the constraint that all observed varaiables are positively correlated with the
        latent variables.
        
        """
        
        super(LinearDecoder_Monotonic, self).__init__()
        self.nCatCovList=nCatCovList
        self.nOutputDim=nOutputDim
        self.dropLinearCovariateColumn=dropLinearCovariateColumn
        self.nLatentDim=nLatentDim
        if len(nCatCovList)>0:
            if dropLinearCovariateColumn:
                self.nInputDimCov=sum(nCatCovList)-len(nCatCovList)
            else:
                self.nInputDimCov=sum(nCatCovList)
            self.linear_cov = nn.Linear(self.nInputDimCov,nOutputDim,bias=False)
        self.linear_latent = LinearPositiveWeights(self.nLatentDim,nOutputDim,bias=False)
        
        
        
    def forward(self,latent_var,*cat_cov_list:int):
        """
        expects a list of categorical covariates (provided as Nx1 dimensional
        vectors). Transforms them into one-hot vectors and returns transformed output.
        """
        one_hot_cat_list = build_onehot_arrays(cat_cov_list,self.nCatCovList,self.dropLinearCovariateColumn)
        latent_component = self.linear_latent(latent_var)
        if len(one_hot_cat_list)>0:
            return self.linear_cov(torch.cat((*one_hot_cat_list,),dim=-1))+latent_component
        else:
            return latent_component


class LinearDecoder(nn.Module):
    def __init__(self,nLatentDim:int,nCatCovList:Iterable[int],nOutputDim:int,dropLinearCovariateColumn:bool):
        """
        Creates a single layer, linear neural network mapping nLatentDim latent variables and sum(nCatCovList) categorical 
        covariates to an output space of nOutputDim. 
        
        """
        
        super(LinearDecoder, self).__init__()
        self.nCatCovList=nCatCovList
        self.nOutputDim=nOutputDim
        self.dropLinearCovariateColumn=dropLinearCovariateColumn
        self.nLatentDim=nLatentDim
        if len(nCatCovList)>0:
            if dropLinearCovariateColumn:
                self.nInputDimCov=sum(nCatCovList)-len(nCatCovList)
            else:
                self.nInputDimCov=sum(nCatCovList)
            self.linear_cov = nn.Linear(self.nInputDimCov,nOutputDim,bias=False)
        self.linear_latent = nn.Linear(self.nLatentDim,nOutputDim,bias=False)
        
        
    def forward(self,latent_var,*cat_cov_list:int):
        """
        expects a list of categorical covariates (provided as Nx1 dimensional
        vectors). Transforms them into one-hot vectors and returns transformed output.
        """
        one_hot_cat_list = build_onehot_arrays(cat_cov_list,self.nCatCovList,self.dropLinearCovariateColumn)
        if len(one_hot_cat_list)>0:
            return self.linear_cov(torch.cat((*one_hot_cat_list,),dim=-1))+self.linear_latent(latent_var)
        else:
            return self.linear_latent(latent_var)
    
class NonlinearMLPDecoder(nn.Module):
    def __init__(self,nLatentDim:int,nCatCovList:Iterable[int],nOutputDim:int,dropLinearCovariateColumn:bool,coupleCovariates:bool,n_layers:int=2,n_hidden:int=128,dropout_rate:float=0.1,use_batch_norm:bool=True):
        super(NonlinearMLPDecoder, self).__init__()
        self.nCatCovList=nCatCovList
        self.nOutputDim=nOutputDim
        self.dropLinearCovariateColumn=dropLinearCovariateColumn
        self.nLatentDim=nLatentDim
        self.coupleCovariates = coupleCovariates
        
        
        if self.coupleCovariates==True:
            self.nonlinear = FCLayers(n_in=self.nLatentDim,n_out=n_hidden,n_cat_list=nCatCovList,n_layers=n_layers,n_hidden=n_hidden,dropout_rate=dropout_rate,use_batch_norm=use_batch_norm)
            self.output_layer =  nn.Linear(n_hidden, self.nOutputDim, bias = False)
        else:
            self.nonlinear_latent = FCLayers(n_in=self.nLatentDim,n_out=n_hidden,n_cat_list=[],n_layers=n_layers,n_hidden=n_hidden,dropout_rate=dropout_rate,use_batch_norm=use_batch_norm)
            self.output_layer =  nn.Linear(n_hidden, self.nOutputDim,bias=False)
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
    def __init__(self,nLatentDim:int,nCatCovList:Iterable[int],nOutputDim:int,dropLinearCovariateColumn:bool,n_layers:int=2,n_hidden:int=128,dropout_rate:float=0.1,use_batch_norm:bool=False):
        super(NonlinearMLPDecoder_Monotonic, self).__init__()
        self.nCatCovList=nCatCovList
        self.nOutputDim=nOutputDim
        self.dropLinearCovariateColumn=dropLinearCovariateColumn
        self.nLatentDim=nLatentDim
        
        if len(nCatCovList)>0:
            if dropLinearCovariateColumn:
                self.nInputDimCov=sum(nCatCovList)-len(nCatCovList)
            else:
                self.nInputDimCov=sum(nCatCovList)
            self.linear_cov = nn.Linear(self.nInputDimCov,nOutputDim,bias=False)
        
        self.nonlinear_pos_latent = FCLayers_Monotonic(n_in=self.nLatentDim,n_out=n_hidden,n_cat_list=[],n_layers=n_layers,n_hidden=n_hidden,dropout_rate=dropout_rate)
        self.output_layer =  LinearPositiveWeights(n_hidden, self.nOutputDim,bias=False)
        
    def forward(self,latent_var,*cat_cov_list:Iterable[int]):
        one_hot_cat_list = build_onehot_arrays(cat_cov_list,self.nCatCovList,self.dropLinearCovariateColumn)
        if len(one_hot_cat_list)>0:
            return self.output_layer(self.nonlinear_pos_latent(latent_var,[]))+self.linear_cov(torch.cat((*one_hot_cat_list,),dim=-1))
        else:
            return self.output_layer(self.nonlinear_pos_latent(latent_var,[]))
                           
if __name__=='__main__':
    n_cov = 0
#    n_cat_cov = [2,3]
    n_cat_cov=[]
    nlatdim = 2
    n_samples = 10
    nOutput = 4
    
    latentVals = torch.randn(n_samples,nlatdim)
    
    #build tensor list
    cat_cov_list=[]
    for n_cat in n_cat_cov:
        cat_cov_list+=[random_catcov(n_cat,n_samples)]
    
    test=NonlinearMLPDecoder(nlatdim,n_cat_cov,nOutput,True,True)
    
    output = test.forward(latentVals,*cat_cov_list)
        