#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:31:35 2019

@author: davidblair
"""

import numpy as np #for testing only
import torch
import torch.nn as nn
from typing import Iterable
from vlpi.FCLayers import FCLayers
from vlpi.utils import random_catcov


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
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

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


if __name__=='__main__':
    n_cov = 5
    n_cat_cov = [2,3,5,10,20]

    n_samples = 100
    nInputDim=500
    nOutputDim=2

    sampledVec = torch.tensor(np.random.binomial(1,0.05,size=(100,500)),dtype=torch.float)

    #build tensor list
    cat_cov_list=[]
    for n_cat in n_cat_cov:
        cat_cov_list+=[random_catcov(n_cat,n_samples)]

#    test=GenericMeanScaleEncoder(n_input=nInputDim, n_output=nOutputDim, n_cat_list=n_cat_cov, n_layers=2,
#                                n_hidden=128, dropout_rate=0.1)
#
#    output = test.forward(sampledVec,*cat_cov_list)

    test=MeanScaleEncoder(n_input=nInputDim,n_output=nOutputDim, n_cat_list=n_cat_cov, n_layers=2,n_hidden=128, dropout_rate=0.1)
    output = test.forward(sampledVec,*cat_cov_list)
