#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:34:37 2019

@author: davidblair
"""
from sklearn.feature_selection import SelectFdr,chi2,f_regression
from vlpi.utils.UtilityFunctions import fisher_exact,T_test
import numpy as np


class FeatureSelection:


    def __init__(self,datasetSampler):
        """
        Class used to perform feature selection on the clinical dataset, passed through using a ClinicalDatasetSampler.

        Parameters
        ----------
        datasetSampler : vlpi.data.ClinicalDatasetSampler
            ClinicalDatasetSampler class for the dataset being analyzed.

        Returns
        -------
        None
        """


        self.sampler=datasetSampler

    def SelectComorbidTraits_ContinuousFeature(self,featureVector,FDR,modifyDataset=False,use_ttest=False):
        """
        Selects features correlated with some continuous variable.

        Parameters
        ----------
        featureVector : [float]
            Vector of floating values for feature selection. Must be sorted in the same order as the index for the ClinicalDatasetSampler training dataset.
        FDR : float
            False discovery rate cut off for feature selection
        modifyDataset : bool
            If True, then features that faile to be selected will be dropped from the dataset.
        use_ttest : bool
            By default, uses F-test to estimate correlation between featureVector and features. If True, instead uses T-test to perform association.

        Returns
        -------
        tuple of arrays
            (Index of selected features, Feature Scores ,Feature P-values)

        """


        previousArrayType = self.sampler.returnArrays
        if self.sampler.returnArrays!='Sparse':
            self.sampler.ChangeArrayType('Sparse')

        sparseTrainingData=self.sampler.ReturnFullTrainingDataset(randomize=False)
        dataMatrix=sparseTrainingData[0]

        if use_ttest:
            fdr=SelectFdr(T_test, alpha=FDR)
        else:
            fdr=SelectFdr(f_regression, alpha=FDR)

        fdr_fit = fdr.fit(dataMatrix,featureVector.ravel())
        discIndx=np.where(fdr_fit.get_support()==True)[0]


        if modifyDataset:
            self.sampler.currentClinicalDataset.IncludeOnly([self.sampler.currentClinicalDataset.dataIndexToDxCodeMap[x] for x in discIndx])

        if previousArrayType!='Sparse':
            self.sampler.ChangeArrayType(previousArrayType)

        return discIndx, fdr_fit.scores_[discIndx],fdr_fit.pvalues_[discIndx]

    def SelectComorbidTraits(self,FDR,modifyDataset=False,useChi2=True):
        """
        Selects features (symptoms) correlated with some dichotomous variable (disease diagnosis), hence co-morbid. This dichotomous variable is automatically inferred from ClinicalDatasetSampler, as it is whatever the sampler is conditioned on.

        Parameters
        ----------

        FDR : float
            False discovery rate cut off for feature selection
        modifyDataset : bool
            If True, then features that faile to be selected will be dropped from the dataset.
        useChi2 : bool
            By default, uses chi-sq test to estimate co-morbidity between featureVector and features. If False, then Fisher's exact test is used. 

        Returns
        -------
        tuple of arrays
            (Index of selected features, Feature Scores ,Feature P-values)

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
