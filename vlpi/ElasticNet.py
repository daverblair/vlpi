#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:08:26 2019

@author: davidblair
"""


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

class ElasticNet:
    
    def __init__(self,num,errorTol,verbose,maxEpochs):
        
        
        self.linMod = LogisticRegression(penalty='elasticnet',tol=errorTol,verbose=verbose,fit_intercept=True,max_iter=maxEpochs,solver='saga',warm_start=True)