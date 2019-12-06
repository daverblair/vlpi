#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:34:37 2019

@author: davidblair
"""
from sklearn.feature_selection import SelectFdr,chi2
from vlpi.utils import fisher_exact
import numpy as np

class FeatureSelection:
    
    
    def __init__(self,datasetSampler):
        """
        Class used to perform feature selection on the clinical dataset, passed through using a ClinicalDatasetSampler
        """
        
        self.sampler=datasetSampler
        
    def SelectComorbidTraits(self,FDR,modifyDataset=False,useChi2=True):
        """
        Selects comorbid traits from the dataset at a false discovery rate of FDR.
        
        datasetSampler: sampler class
        FDR: false discovery rate
        modifyDataset: indicates whether to modify datset such that only comorbid terms are included in further analyses.
        useChi2: whether chi2 (default) should be used for comorbidity. Alternative is fisher's exact test, which is slower
        
        """
        
        
        assert self.sampler.isConditioned==True,"Cannot perform feature selection without being conditioned on some disease of interest"
        previousArrayType = self.sampler.returnArrays
        if self.sampler.returnArrays!='Sparse':
            self.sampler.ChangeArrayType('Sparse')
        
            
        sparseTrainingData=self.sampler.ReturnFullTrainingDataset(randomize=False)
        dataMatrix=sparseTrainingData[0]
        incidenceVec =sparseTrainingData[2]
        
        if useChi2==False:
            fdr=SelectFdr(fisher_exact, alpha=FDR)
        else:
            fdr=SelectFdr(chi2, alpha=FDR)
            
            
        fdr_fit = fdr.fit(dataMatrix,incidenceVec.toarray())
        discIndx=np.where(fdr_fit.get_support()==True)[0]
        
        if modifyDataset:
            self.sampler.currentClinicalDataset.IncludeOnly([self.sampler.currentClinicalDataset.dataIndexToDxCodeMap[x] for x in discIndx])
        
        if previousArrayType!='Sparse':
            self.sampler.ChangeArrayType(previousArrayType)
        
        return discIndx, fdr_fit.scores_[discIndx],fdr_fit.pvalues_[discIndx]