#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:13:55 2019

@author: davidblair
"""
import torch
import numpy as np
import subprocess
from scipy.stats import beta,norm
from scipy.optimize import fsolve
from scipy import sparse
from scipy.stats import fisher_exact as _fisher_exact

def build_onehot_arrays(cat_cov_list,nCatCovList,dropOneColumn):
    """
    Builds one-hot arrays from integer categories. Adapted from scVI.
    cat_cov_list: list of categorical arrays (torch.tensors)
    nCatCovList: list indicating number of categories for each covariate
    dropOneColumn: indicates whether or not to drop one column from the coviariates, eliminates colinearity

    """

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
    
def one_hot_scipy(index,n_cat,dropColumn=False):
    if sparse.issparse(index):
        index = index.toarray().ravel()
    else:
        index=np.array(index).ravel()
    one_hot = sparse.coo_matrix((np.ones(index.shape[0]),(np.arange(index.shape[0]),index)),shape = (index.shape[0],n_cat),dtype=np.float64)
    one_hot = one_hot.tocsr()
    if dropColumn:
        one_hot = one_hot[:,1:]
    return one_hot

def file_len(fname,skip_rows):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])-skip_rows

def random_catcov(n_cat,n_samples,device):
    samp_params = np.random.dirichlet(np.ones((n_cat)))
    return torch.tensor(np.random.choice(np.arange(n_cat),size=(n_samples,1),p=samp_params),device=device)

def _beta_CI_zero_func(params,CIvals,CIlevel):
    assert CIlevel<1.0 and CIlevel>0.0,"CI must lie between 0.0 and 1.0"
    p=(1.0-CIlevel)/2.0
    return (beta.ppf(p,params[0],params[1])-CIvals[0],beta.ppf(CIlevel+p,params[0],params[1])-CIvals[1])

def infer_beta_from_CI(CI95):
    initParam = np.array([1.0,1.0])
    output = fsolve(_beta_CI_zero_func,initParam,args=(CI95,0.95),full_output=True)
    assert output[2]==1,"Unable to find beta distribution with appropriate CI, consider alternative."
    return output[0]

def _liability_CI_zero_func(params, CIvals,CIlevel):
    assert CIlevel<1.0 and CIlevel>0.0,"CI must lie between 0.0 and 1.0"
    p = (1.0-CIlevel)/2.0
    cDist = norm(params[0],params[1])
    return (norm(0.0,1.0).cdf(cDist.ppf(p))-CIvals[0],norm(0.0,1.0).cdf(cDist.ppf(CIlevel+p))-CIvals[1])
        

def infer_liability_CI(CI95):
    initParam = np.zeros(2)
    initParam[0] = np.mean(norm(0.0,1.0).ppf(CI95))
    initParam[1] = 1e-6
    output = fsolve(_liability_CI_zero_func,initParam,args=(CI95,0.95),full_output=True)
    assert output[2]==1,"Unable to find gaussian distribution with appropriate CI, consider alternative."
    return output[0]



def fisher_exact(dataMatrix,incidenceVec):
    """
    dataMatrix: sparse NxK matrix of binary features
    incidenceVec: sparse Nx1 vector of binary labels

    returns: score,pval
    """

    numLabel = incidenceVec.sum()
    whereLabelTrue = incidenceVec.nonzero()[0]
    numTotal = dataMatrix.shape[0]
    scores = np.zeros(dataMatrix.shape[1],dtype=np.float64)
    pvals = np.zeros(dataMatrix.shape[1],dtype=np.float64)

    for feature_i in range(dataMatrix.shape[1]):
        numFeature = dataMatrix[:,feature_i].sum()
        numBoth = dataMatrix[whereLabelTrue,feature_i].sum()
        numFeatureOnly = numFeature - numBoth
        numLabelOnly = numLabel - numBoth
        numNeither = numTotal - numBoth-numLabelOnly-numFeatureOnly
        fisher_test_results = _fisher_exact(np.array([[numBoth,numLabelOnly],[numFeatureOnly,numNeither]]))
        scores[feature_i]=fisher_test_results[0]
        pvals[feature_i]=fisher_test_results[1]
    return scores,pvals


if __name__=='__main__':
    c = np.array([0.0005,0.0015])
    
    prior=np.array([1.0,1.0])
    lambda_param = np.random.gamma(prior[0],1.0/prior[1])
    obs_data = np.random.exponential(1.0/np.sqrt(lambda_param),1000)

    lambda_inf = infer_gamma_params_linear_monotonic(obs_data,prior)