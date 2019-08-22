#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:10:20 2019

@author: davidblair
"""

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np

class Optimizers:
    
    def __init__(self,phenotypeModel,datasetSampler, learningRate = 1e-3, slidingErrorWindow = 10):
        """
        This class implements two SGD optimizers, one of which uses the full training
        dataset (FullDatasetTrain) and the other uses mini-batches randomly sampled
        from the training dataset.
        
        phenotypeModel-->The vLPM model to optimize
        datasetSampler-->ClinicalDatasetSampler that holds clinical data and generates random samples
        learningRate-->learningRate for the ADAM SGD optimizer
        slidingErrorWindow-->sliding window over which the ELBO is averaged over. Default is 10 epochs.
        """
        
        self.phenotypeModel=phenotypeModel
        self.datasetSampler=datasetSampler
        self.optimizer = Adam({"lr": learningRate}) 
        self.svi = SVI(self.phenotypeModel.model, self.phenotypeModel.guide, self.optimizer, loss=Trace_ELBO())
        self.errorWindow=slidingErrorWindow
        
        
        
    def FullDatasetTrain(self,maxEpochs, errorTol = 1e-5,verbose=True):
        """
        maxEpochs-->maximum number of training epochs.
        errorTol--> when error in avg ELBO < errorTol, optimization stops.
        verbose--> if True, prints out training ELBO, test ELBO, and error at the
        end of every slidingErrorWindow
        """
        
        errorVec =[]
        runningAvgTrainLoss = []
        runningAvgTestLoss = []
        currentEpochWindowTrainLoss=0.0
        currentEpochWindowTestLoss=0.0
        
        self.phenotypeModel._setTrainMode()
        for epoch in range(maxEpochs):
            
            
            trainData=self.datasetSampler.ReturnFullTrainingDataset()
            testData=self.datasetSampler.ReturnFullTestingDataset()
            currentEpochWindowTrainLoss+=self.svi.step(*trainData)
            currentEpochWindowTestLoss+=self.svi.evaluate_loss(*testData)
            
            currentWindow,leftover  = divmod(epoch+1,self.errorWindow)
            if leftover==0:
                currentEpochWindowTrainLoss = currentEpochWindowTrainLoss/self.errorWindow
                currentEpochWindowTestLoss = currentEpochWindowTestLoss/self.errorWindow
                if currentWindow>1:
                    runningAvgTrainLoss+=[(runningAvgTrainLoss[-1]*(currentWindow-1)+currentEpochWindowTrainLoss)/currentWindow]
                    runningAvgTestLoss+=[(runningAvgTestLoss[-1]*(currentWindow-1)+currentEpochWindowTestLoss)/currentWindow]
                    errorVec+=[np.abs((runningAvgTestLoss[-1]-runningAvgTestLoss[-2])/runningAvgTestLoss[-1])]
                else: 
                    runningAvgTrainLoss+=[currentEpochWindowTrainLoss]
                    runningAvgTestLoss+=[currentEpochWindowTestLoss]
                    errorVec+=[1.0]
                currentEpochWindowTrainLoss=0.0
                currentEpochWindowTestLoss=0.0
                if verbose:
                    print("Completed %03d epochs; Current Avg Loss (Train, Test): %.4f, %.4f; Error: %.4e"%(epoch+1,runningAvgTrainLoss[-1],runningAvgTestLoss[-1],errorVec[-1]))
                if errorVec[-1] < errorTol:
                    break
                
        self.phenotypeModel._setEvalMode()
        return runningAvgTrainLoss,runningAvgTestLoss,errorVec
    
    def BatchTrain(self,maxEpochs,batch_size,errorTol = 1e-3,verbose=True):
        """
        maxEpochs-->maximum number of training epochs.
        errorTol--> when error in avg ELBO < errorTol, optimization stops.
        verbose--> if True, prints out training ELBO, test ELBO, and error at the
        end of every slidingErrorWindow
        """
        errorVec=[]
        runningAvgTrainLoss = []
        runningAvgTestLoss = []
        currentEpochWindowTrainLoss=0.0
        currentEpochWindowTestLoss=0.0
        
        if isinstance(self.datasetSampler.trainingDataIndex,list):
            numTotalTrainingSamples=sum([len(x) for x in self.datasetSampler.trainingDataIndex])
        else:
            numTotalTrainingSamples=len(self.datasetSampler.trainingDataIndex)
        if isinstance(self.datasetSampler.testDataIndex,list):
            numTotalTestSamples=sum([len(x) for x in self.datasetSampler.testDataIndex])
        else:
            numTotalTestSamples=len(self.datasetSampler.testDataIndex)
        
        self.phenotypeModel._setTrainMode()
        for epoch in range(maxEpochs):
            avg_epoch_loss = 0.0
            numBatches=0
            for data_batch in self.datasetSampler.TrainingEpochGenerator(batch_size):
                avg_epoch_loss+=self.svi.step(*data_batch)*(numTotalTrainingSamples/data_batch[0].shape[0])
                numBatches+=1
            currentEpochWindowTrainLoss+=avg_epoch_loss/numBatches
            
            avg_epoch_loss = 0.0
            numBatches=0
            for data_batch in self.datasetSampler.TestingEpochGenerator(batch_size):
                avg_epoch_loss+=self.svi.evaluate_loss(*data_batch)*(numTotalTestSamples/data_batch[0].shape[0])
                numBatches+=1
            currentEpochWindowTestLoss+=avg_epoch_loss/numBatches
            
            currentWindow,leftover  = divmod(epoch+1,self.errorWindow)
            if leftover==0:
                currentEpochWindowTrainLoss = currentEpochWindowTrainLoss/self.errorWindow
                currentEpochWindowTestLoss = currentEpochWindowTestLoss/self.errorWindow
                if currentWindow>1:
                    runningAvgTrainLoss+=[(runningAvgTrainLoss[-1]*(currentWindow-1)+currentEpochWindowTrainLoss)/currentWindow]
                    runningAvgTestLoss+=[(runningAvgTestLoss[-1]*(currentWindow-1)+currentEpochWindowTestLoss)/currentWindow]
                    errorVec+=[np.abs((runningAvgTestLoss[-1]-runningAvgTestLoss[-2])/runningAvgTestLoss[-1])]
                else: 
                    runningAvgTrainLoss+=[currentEpochWindowTrainLoss]
                    runningAvgTestLoss+=[currentEpochWindowTestLoss]
                    errorVec+=[1.0]
                currentEpochWindowTrainLoss=0.0
                currentEpochWindowTestLoss=0.0
                if verbose:
                    print("Completed %03d epochs; Current Avg Loss (Train, Test): %.4f, %.4f; Error: %.4e"%(epoch+1,runningAvgTrainLoss[-1],runningAvgTestLoss[-1],errorVec[-1]))
                if errorVec[-1] < errorTol:
                    break
        self.phenotypeModel._setEvalMode()
        return runningAvgTrainLoss,runningAvgTestLoss,errorVec
    