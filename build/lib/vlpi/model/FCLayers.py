#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:59:06 2019

@author: davidblair
"""

import collections
from typing import Iterable
import torch
from torch import nn as nn
from vlpi.utils.UtilityFunctions import one_hot
from vlpi.utils.LinearPositiveWeights import LinearPositiveWeights


class FCLayers_Monotonic(nn.Module):
    """A helper class to build fully-connected layers for a neural network.
    Uses Tanh for non-linearities; dropoutfor convergence.
    Adapted frome the scVI package: https://github.com/YosefLab/scVI.

    :param n_in: The dimensionality of the input data (excluding covariates)
    :param n_out: The dimensionality of the output layer
    :param n_cat_list: A list containing, for each categorical covariate of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    Note: there no batch normalization with this module, as this can break the
    positive correlation structure desired by the model. Moreover, we replaced
    the ReLU function with tanh function which ensures monotonicity without erasing
    negative values (which the ReLU will do by design)
    """
    def __init__(self, n_in: int, n_out: int, n_cat_list: Iterable[int] = None,
                 n_layers: int = 2, n_hidden: int = 128, dropout_rate: float = 0.2, use_batch_norm=True):

        super(FCLayers_Monotonic,self).__init__()

        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(collections.OrderedDict(
            [('Layer {}'.format(i), nn.Sequential(
                LinearPositiveWeights(n_in + sum(self.n_cat_list), n_out,bias=True),
                nn.Tanh(),
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None))
             for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))]))

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward computation on ``x``.
        :param x: tensor of values with shape ``(n_samples,n_in)``
        :param cat_list: list of category membership(s) for this sample, provided as (1,n_sample) tensors
        :return: tensor of shape ``(n_out,n_samples)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(cat_list), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (n_cat and cat is None), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, LinearPositiveWeights):
                        if x.size(0)!=0:
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        else:
                            x = torch.cat((*one_hot_cat_list,), dim=-1)

                    x = layer(x)
        return x


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.
    Uses ReLU for non-linearities, BatchNorm1d and dropout for convergence.
    Adapted frome the scVI package: https://github.com/YosefLab/scVI.

    :param n_in: The dimensionality of the input data (excluding covariates)
    :param n_out: The dimensionality of the output layer
    :param n_cat_list: A list containing, for each categorical covariate of interest,
                 the number of categories. Each category will be
                 included using a one-hot encoding.
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_batch_norm: whether or not to use batch normalization to speed convergence
    """


    def __init__(self, n_in: int, n_out: int, n_cat_list: Iterable[int] = None,
                 n_layers: int = 1, n_hidden: int = 128, dropout_rate: float = 0.2, use_batch_norm=True):

        super(FCLayers,self).__init__()

        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        self.fc_layers = nn.Sequential(collections.OrderedDict(
            [('Layer {}'.format(i), nn.Sequential(
                nn.Linear(n_in + sum(self.n_cat_list), n_out,bias=True),
                nn.BatchNorm1d(n_out) if use_batch_norm else None,
                nn.ReLU(),
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None))
             for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))]))

    def forward(self, x: torch.Tensor, *cat_list: int):
        """Forward computation on ``x``.
        :param x: tensor of values with shape ``(n_samples,n_in)``
        :param cat_list: list of category membership(s) for this sample, provided as (1,n_sample) tensors
        :return: tensor of shape ``(n_out,n_samples)``
        :rtype: :py:class:`torch.Tensor`
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.
        assert len(self.n_cat_list) <= len(cat_list), "nb. categorical args provided doesn't match init. params."
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (n_cat and cat is None), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for layers in self.fc_layers:
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.Linear):
                        if x.size(0)!=0:
                            x = torch.cat((x, *one_hot_cat_list), dim=-1)
                        else:
                            x = torch.cat((*one_hot_cat_list,), dim=-1)
                    x = layer(x)
        return x
