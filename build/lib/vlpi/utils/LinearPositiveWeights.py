#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:51:17 2019

@author: davidblair
"""

import torch
import torch.nn as nn
import math

class LinearPositiveWeights(nn.Module):

    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        """
        Modifies the Linear neural network class such that the weights are always positive,
        ensuring outputs are a monotonic function of inputs.
        
        """
        super(LinearPositiveWeights, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_scale_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Uses exponential distribution to initialize network weights.
        """
        
        with torch.no_grad():
            self.log_scale_weight.exponential_(100.0)
            self.log_scale_weight=self.log_scale_weight.log_()
            
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.nn.functional.softplus(self.log_scale_weight))
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return nn.functional.linear(input, torch.nn.functional.softplus(self.log_scale_weight), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
