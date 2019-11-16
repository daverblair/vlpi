#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:31:35 2019

@author: davidblair
"""

import torch
import torch.nn as nn
from typing import Iterable
from vlpi.FCLayers import FCLayers
import collections
import torch.distributions as dist


class MeanScaleEncoder(nn.Module):
    r"""
    Adapted from the scVI module: https://github.com/YosefLab/scVI


    Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int,
                n_cat_list: Iterable[int] = None, n_layers: int = 2,
                 n_hidden: int = 128, dropout_rate: float = 0.1, use_batch_norm:bool=True):

        super(MeanScaleEncoder,self).__init__()

        self.encoder = FCLayers(n_in=n_input, n_out=n_hidden, n_cat_list=n_cat_list, n_layers=n_layers,
                                n_hidden=n_hidden, dropout_rate=dropout_rate,use_batch_norm=use_batch_norm)
        self.mean_encoder = nn.Linear(n_hidden, n_output,bias=True)
        self.var_encoder = nn.Linear(n_hidden, n_output,bias=True)

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-6
        return q_m, q_v
    
class BinaryExpectationEncoder(nn.Module):
    r"""
    Adapted from the scVI module: https://github.com/YosefLab/scVI


    Encodes data of ``n_input`` dimensions into a binary latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int,
                n_cat_list: Iterable[int] = None, n_layers: int = 2,
                 n_hidden: int = 128, dropout_rate: float = 0.1, use_batch_norm:bool=True):

        super(BinaryExpectationEncoder,self).__init__()

        self.encoder = FCLayers(n_in=n_input, n_out=n_hidden, n_cat_list=n_cat_list, n_layers=n_layers,
                                n_hidden=n_hidden, dropout_rate=dropout_rate,use_batch_norm=use_batch_norm)
        
        self.bin_exp_encoder = nn.Sequential(collections.OrderedDict([('Linear',nn.Linear(n_hidden, n_output,bias=True)),('Sigmoid',nn.Sigmoid())]))
        

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        """

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        bin_exp = self.bin_exp_encoder(q)
        return torch.cat((1.0-bin_exp,bin_exp),dim=-1)

    
    
            
        
        
    
if __name__=='__main__':
    import torch.distributions as dist
    nInput = 100
    nSample = 5
    nOutput = 5
    nOutput=2
    
#    inputData = dist.Bernoulli(0.2).sample((nSample,nInput))
#    tmp=BinaryExpectationEncoder(nInput,nOutput)

    scores = dist.Normal(0.0,1.0).sample([nInput,1])
    
    tmp = RankEncoder_Binary(0.05)
    