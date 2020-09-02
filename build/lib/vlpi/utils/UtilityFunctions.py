#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:13:55 2019

@author: davidblair
"""
import torch
import numpy as np
import subprocess
from scipy import sparse
from scipy.stats import fisher_exact as _fisher_exact
from scipy.stats import ttest_ind

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





def rel_diff(curr,prev):
    return abs((curr - prev) / prev)


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


def T_test(dataMatrix,scoreVec):
    scores = np.zeros(dataMatrix.shape[1],dtype=np.float64)
    pvals = np.zeros(dataMatrix.shape[1],dtype=np.float64)
    for feature_i in range(dataMatrix.shape[1]):
        where_nonzero=dataMatrix[:,feature_i].nonzero()[0]
        where_zero = np.setdiff1d(np.arange(dataMatrix.shape[0]),where_nonzero)

        pop_a = scoreVec[where_nonzero]
        pop_b = scoreVec[where_zero]

        stats = ttest_ind(pop_a, pop_b,equal_var=False)
        scores[feature_i]=stats[0]
        pvals[feature_i]=stats[1]
    return scores,pvals
