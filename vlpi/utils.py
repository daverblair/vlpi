#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:13:55 2019

@author: davidblair
"""
import torch
import numpy as np
import subprocess
from scipy.stats import beta
from scipy.optimize import fsolve

def build_onehot_arrays(cat_cov_list,nCatCovList,dropOneColumn):
    one_hot_cat_list = []  # for generality in this list many indices useless.
    assert len(nCatCovList) == len(cat_cov_list), "number of categorical args provided doesn't match initial params."
    for n_cat, cat in zip(nCatCovList, cat_cov_list):
        assert not (n_cat and cat is None), "category not provided while n_cat != 0 in init. params."
        if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
            if cat.size(1) != n_cat:
                one_hot_cat = one_hot(cat, n_cat,dropOneColumn)
            else:
                one_hot_cat = cat  # cat has already been one_hot encoded
            one_hot_cat_list += [one_hot_cat]
    return one_hot_cat_list

def one_hot(index, n_cat, dropColumn=False):
    """
    expects tensor of shape (n_samples,1), returns one-hot array size 
    (n_samples, n_cat). 
    
    Optionally, can drop the first column of the encoding to prevent colinearity
    among the predictors. This is only necessary if you care about the values
    of the inferred parameters during inference.
    """
        
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    if dropColumn:
        return onehot.type(torch.float32)[:,1:]
    else:
        return onehot.type(torch.float32)
    
    
def file_len(fname,skip_rows):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])-skip_rows

def random_catcov(n_cat,n_samples):
    samp_params = np.random.dirichlet(np.ones((n_cat)))
    return torch.tensor(np.random.choice(np.arange(n_cat),size=(n_samples,1),p=samp_params))

def _beta_CI_zero_func(params,CIvals,CIlevel):
    assert CIlevel<1.0 and CIlevel>0.0,"CI must lie between 0.0 and 1.0"
    p=(1.0-CIlevel)/2.0
    return (beta.ppf(p,params[0],params[1])-CIvals[0],beta.ppf(CIlevel+p,params[0],params[1])-CIvals[1])

def infer_beta_from_CI(CI95):
    initParam = np.array([1.0,1.0])
    output = fsolve(_beta_CI_zero_func,initParam,args=(CI95,0.95),full_output=True)
    assert output[2]==1,"Unable to find beta distribution with appropriate CI, consider alternative."
    return output[0]


if __name__=='__main__':
    c = np.array([0.0001,0.001])
    betaParams = infer_beta_from_CI(c)
    
    

    
