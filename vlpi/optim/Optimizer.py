#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:10:20 2019

@author: davidblair
"""
import torch
import pyro
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from torch.utils import data
from pyro.optim import CosineAnnealingWarmRestarts,ReduceLROnPlateau
from collections import deque
import numpy as np
from scipy.stats import linregress
from pyro import poutine
from vlpi.optim.AdamW import AdamW
from vlpi.data.ClinicalDataset import _TorchDatasetWrapper
from vlpi.utils.UtilityFunctions import rel_diff


class Optimizer:
        
    def _returnELBOSlope(self,elbo_deque):
        return abs(linregress(range(len(elbo_deque)),elbo_deque)[0])/np.mean(elbo_deque)
        
    def _localShuffle(self,dataSample):
        """
        shuffles dataSample locally on disk/gpu
        """
        
        if self.useCuda:
            new_index = torch.randperm(dataSample[0].shape[0],device = self.device)
            old_index = torch.arange(dataSample[0].shape[0],device = self.device)
        else:
            new_index = torch.randperm(dataSample[0].shape[0])
            old_index = torch.arange(dataSample[0].shape[0])
            
        dataSample[0][old_index] = dataSample[0][new_index]
        for i in range(len(dataSample[1])):
            dataSample[1][i][old_index]=dataSample[1][i][new_index]
            
        if dataSample[2] is not None:
            dataSample[2][old_index]=dataSample[2][new_index]
        if dataSample[3] is not None:
            dataSample[3][old_index]=dataSample[3][new_index]
            
            
    def _sendDataToGPU(self,returnData):
        """
        sends returnData to gpu
        """
        
        newIncData=returnData[0].to('cuda:{}'.format(self.device))
        newCovData=[]
        for i,arr in enumerate(returnData[1]):
           newCovData+=[arr.to('cuda:{}'.format(self.device))]
           
        if returnData[2] is not None:
            newTargetData=returnData[2].to('cuda:{}'.format(self.device))
        else:
            newTargetData=None
        if returnData[3] is not None:
            newScoreData=returnData[3].to('cuda:{}'.format(self.device))
        else:
            newScoreData=None
            
        return (newIncData,newCovData,newTargetData,newScoreData)
        
        

    def __init__(self,phenotypeModel,datasetSampler,optimizationParameters={'initialLearningRate': 0.05,'maxEpochs': 5000,'numParticles':10},computeConfiguration={'device':None,'numDataLoaders':0},cosineAnnealing={'withCosineAnnealing':False,'initialRestartInterval':10,'intervalMultiplier':2}, **kwargs):
        """
        This class implements two SGD optimizers, one of which uses the full training
        dataset (FullDatasetTrain) and the other uses mini-batches randomly sampled
        from the training dataset. To improve convergence rate and escape local minima, 
        learning rate can be autommatically altered during inference using
        Cosine Annealing with Warm Restarts (see Loshchilov & Hutter 2017).
        
        Note, given stochastic inference strategy coupled with possible re-starts,
        class tracks model perfomance and stores the best possible instance of the model
        obtained throughout the inference process. This also helps to avoid overfitting. 

        phenotypeModel-->The phenotype model to optimize
        datasetSampler-->ClinicalDatasetSampler that holds clinical data and generates random samples
        withCosineAnnealing-->whether to use Cosine Annealing to adjust learning rate during inference. Defaults to scheduler = ReduceLROnPlateau otherwise.
 
        maxLearningRate-->maximum learning rate used by SGD during inference. If withCosineAnnealing = False, then optimization is initiated with this learning rate once, which is adjusted per LR reduction on plateau with faily strict reduction policy
        device-->compute device to use. By default, uses CPU unless GPU device specified, which can be string or integer
    
        """

        allKeywordArgs = list(kwargs.keys()) 
        
        self.phenotypeModel=phenotypeModel
        self.datasetSampler=datasetSampler
        
        #general optimization parameters
        self.maxEpochs=optimizationParameters['maxEpochs']
        self.learningRate = optimizationParameters['initialLearningRate']
        self.numParticles = optimizationParameters['numParticles']
        
        #compute resources parameters
        self.device=computeConfiguration['device']
        if self.device is None:
            self.device='cpu'
            
        if self.device!='cpu':
            assert torch.cuda.is_available()==True, 'GPU is not available on your system.'
            self.useCuda=True
            if isinstance(self.device,str):
                self.device=int(self.device.strip('cuda:'))
            
        else:
            self.useCuda=False

        self.num_dataloaders=computeConfiguration['numDataLoaders']
        
        #cosine annealing parameters
        
        self.withCosineAnnealing = cosineAnnealing['withCosineAnnealing']
        self.cosineAnnealingParameters = [cosineAnnealing['initialRestartInterval'],cosineAnnealing['intervalMultiplier']]
        
        if 'AdamW_weight_decay' in allKeywordArgs:
            self.AdamW_weight_decay=kwargs['AdamW_weight_decay']
        else:
            self.AdamW_weight_decay=1e-4
            
        if 'LRPlateauTol' in allKeywordArgs:
            self.LRPlateauTol=kwargs['LRPlateauTol']
        else:
            self.LRPlateauTol=0.01
        

    def _contstructSVI(self,optimizationStrategy):
        """
        Constructs a list of SVI functions use to train model. Takes optimizationStrategy as input, which must be in ['Full','PosteriorOnly','GuideOnly']
        """
        
        
        
        if optimizationStrategy=='ScoreBased':
            _model=self.phenotypeModel._encoder_only_model
            _guide=self.phenotypeModel._encoder_only_guide
        elif optimizationStrategy=='Full':
            _model=self.phenotypeModel.model
            _guide=self.phenotypeModel.guide
        
        else:
            test_data = self.datasetSampler.GenerateRandomTrainingSample(2)
            if self.useCuda:
                test_data=self._sendDataToGPU(test_data)
        
            self.phenotypeModel.guide(*test_data)
            param_store=pyro.get_param_store()
            if optimizationStrategy=='DecoderOnly':
                _model=self.phenotypeModel.model
                _guide=poutine.block(self.phenotypeModel.guide,hide=[x for x in param_store.get_all_param_names() if ('encoder' in x)])
            
            else:
                _model=poutine.block(self.phenotypeModel.model,hide=[x for x in param_store.get_all_param_names() if ('decoder' in x)])
                _guide=self.phenotypeModel.guide
            
        
        
            
        #initialize the SVI class
        AdamWOptimArgs = {'lr': self.learningRate,'weight_decay':self.AdamW_weight_decay}
        if self.withCosineAnnealing:
            scheduler = CosineAnnealingWarmRestarts({'optimizer': AdamW, 'optim_args': AdamWOptimArgs, 'T_0': self.cosineAnnealingParameters[0], 'T_mult': self.cosineAnnealingParameters[1]})
            return {'svi':SVI(_model,_guide,scheduler, loss=Trace_ELBO(num_particles=self.numParticles)),'scheduler':scheduler}
                
        else:  
            scheduler = ReduceLROnPlateau({'optimizer': AdamW, 'optim_args': AdamWOptimArgs, 'threshold': self.LRPlateauTol})
            return {'svi':SVI(_model,_guide,scheduler, loss=Trace_ELBO(num_particles=self.numParticles)),'scheduler':scheduler}
        
        

    def FullDatasetTrain(self, errorTol:float = 1e-5,verbose:bool=True,logFile=None,optimizationStrategy='Full',errorComputationWindow=None):
        """
        maxEpochs-->maximum number of training epochs.
        errorTol--> when error in avg testing ELBO < errorTol, optimization stops.
        verbose--> if True, prints out training ELBO, test ELBO, and error at the
        end of every slidingErrorWindow
        logFile-->prints out training information to logFile
        """
        if errorComputationWindow is None:
            errorComputationWindow = max(int(0.1*self.maxEpochs),2)
        else:
            assert errorComputationWindow>0, "Expects errorComputationWindow to be integer > 0"
            
        error_window = deque(maxlen=errorComputationWindow)
        elbo_window = deque(maxlen=errorComputationWindow)
        errorVec = []
        trainLoss = []
        testLoss = []
        
        
        
        if self.useCuda:
            self.phenotypeModel.SwitchDevice(self.device)
        self.phenotypeModel.train()
        
        #training and test data are fixed for full dataset training; therefore, just need
        #load into memory once.
        if self.useCuda:
            trainData=self._sendDataToGPU(self.datasetSampler.ReturnFullTrainingDataset())
            testData=self._sendDataToGPU(self.datasetSampler.ReturnFullTestingDataset())
        else:
            trainData=self.datasetSampler.ReturnFullTrainingDataset()
            testData=self.datasetSampler.ReturnFullTestingDataset()
        
        
        
        assert optimizationStrategy in ['ScoreBased','Full','EncoderOnly','DecoderOnly'],"Available Optimization strategies: ['ScoreBased','Full','EncoderOnly','DecoderOnly']"
        sviFunction = self._contstructSVI(optimizationStrategy)
        
        sviFunction = self._contstructSVI(optimizationStrategy)
        
        # initialize model and parameter storage vectors. Note, parameter storage vectors are stored in a list with each element corresponding to one SVI function
        bestModelState= self.phenotypeModel.PackageCurrentState()
        bestModelScore = sviFunction['svi'].evaluate_loss(*testData)
        prev_train_loss = sviFunction['svi'].evaluate_loss(*trainData)
    
        
        for epoch in range(self.maxEpochs):
            self._localShuffle(trainData)
            
            # take step on training data
            currrent_train_loss=sviFunction['svi'].step(*trainData)
                        
            #adjust learning rate per scheduler
            if self.withCosineAnnealing:
                sviFunction['scheduler'].step(epoch=epoch)
            else:
                sviFunction['scheduler'].step(currrent_train_loss)
            
            
            #no point in shuffling testing data, all observations are independent
            #also, all svi functions should return identical loss, so we arbitrarily choose the latter for loss
            current_test_loss=sviFunction['svi'].evaluate_loss(*testData)
            
            if current_test_loss<bestModelScore:
                bestModelState = self.phenotypeModel.PackageCurrentState()
                bestModelScore = current_test_loss
        
            #track overall convergence
            trainLoss+=[currrent_train_loss]
            testLoss+=[current_test_loss]
            
            error_window.append(rel_diff(currrent_train_loss,prev_train_loss))
            elbo_window.append(currrent_train_loss)
            
            avg_error = sum(error_window)/len(error_window)
            med_error = sorted(error_window)[int(0.5*len(error_window))]
            slope_error = self._returnELBOSlope(elbo_window)
            
            prev_train_loss=currrent_train_loss
            errorVec+=[min([avg_error,med_error,slope_error])]
            
            if (avg_error < errorTol) or (med_error < errorTol) or (slope_error<errorTol):
                

                if verbose:
                    print("Optimization converged in %03d epochs; Current Loss (Train, Test): %.4f, %.4f; Error: %.4e"%(epoch+1,trainLoss[-1],testLoss[-1],min([avg_error,med_error,slope_error])))
                if logFile!=None:
                    with open(logFile, "a") as f:
                        f.write("Optimization converged in %03d epochs; Current Loss (Train, Test): %.4f, %.4f; Error: %.4e\n"%(epoch+1,trainLoss[-1],testLoss[-1],min([avg_error,med_error,slope_error])))
                break
            
            else:
                if verbose:
                    print("Completed %03d epochs; Current Loss (Train, Test): %.4f, %.4f; Error: %.4e"%(epoch+1,trainLoss[-1],testLoss[-1],min([avg_error,med_error,slope_error])))
                if logFile!=None:
                    with open(logFile, "a") as f:
                        f.write("Completed %03d epochs; Current Loss (Train, Test): %.4f, %.4f; Error: %.4e\n"%(epoch+1,trainLoss[-1],testLoss[-1],errorVec[-1][0],min([avg_error,med_error,slope_error])))
            
                
        if epoch == (self.maxEpochs-1):
            print('Warning: Optimization did not converge in allotted epochs')
        
        self.phenotypeModel.LoadPriorState(bestModelState)
        #set eval mode
        if self.useCuda:
            self.phenotypeModel.SwitchDevice('cpu')
        self.phenotypeModel.eval()
        return bestModelScore,trainLoss,testLoss,errorVec

    def BatchTrain(self,batch_size:int,errorTol:float = 1e-3,verbose=True,logFile=None,optimizationStrategy='Full',errorComputationWindow=None):
        """
        maxEpochs-->maximum number of training epochs.
        errorTol--> when error in avg testg ELBO < errorTol, optimization stops.
        verbose--> if True, prints out training ELBO, test ELBO, and error at the
        end of every slidingErrorWindow
        """
        if errorComputationWindow is None:
            errorComputationWindow = max(int(0.05*self.maxEpochs),2)
        else:
            assert errorComputationWindow>0, "Expects errorComputationWindow to be integer > 0"
        
        error_window = deque(maxlen=errorComputationWindow)
        elbo_window = deque(maxlen=errorComputationWindow)        
        errorVec = []
        trainLoss = []
        testLoss = []

        if isinstance(self.datasetSampler.trainingDataIndex,list):
            numTotalTrainingSamples=sum([len(x) for x in self.datasetSampler.trainingDataIndex])
            numTotalTestingSamples=sum([len(x) for x in self.datasetSampler.testDataIndex])
        else:
            numTotalTrainingSamples=len(self.datasetSampler.trainingDataIndex)
            numTotalTestingSamples=len(self.datasetSampler.testDataIndex)

        assert batch_size < numTotalTrainingSamples, "Batch size is greater than or equal to training data size. Use FullDatasetTrain"
        
                

        if self.useCuda:
            self.phenotypeModel.SwitchDevice(self.device)
            
        self.phenotypeModel.train()

        
        # for batch training, use PyTorch dataset loaders, as they allow threaded queuing of samples, which can greatly improve performance.
        torchTrainingData=_TorchDatasetWrapper(self.datasetSampler,self.datasetSampler.trainingDataIndex,batch_size)
        torchTestingData=_TorchDatasetWrapper(self.datasetSampler,self.datasetSampler.testDataIndex,batch_size)
        
        #note, batch_size handled at the level of _TorchDatasetWrapper, not loader itself.
        trainingDataLoader = data.DataLoader(torchTrainingData,batch_size=1,shuffle=False,sampler=None,batch_sampler=None,num_workers=self.num_dataloaders,collate_fn=lambda x:x[0],pin_memory=self.useCuda)
        testingDataLoader = data.DataLoader(torchTestingData,batch_size=1,shuffle=False,sampler=None,batch_sampler=None,num_workers=self.num_dataloaders,collate_fn=lambda x:x[0],pin_memory=self.useCuda)
        
        assert optimizationStrategy in ['ScoreBased','Full','EncoderOnly','DecoderOnly'],"Available Optimization strategies: ['ScoreBased','Full','EncoderOnly','DecoderOnly']"
        sviFunction = self._contstructSVI(optimizationStrategy)
        
        
        #initialize model states and scores
        bestModelState= self.phenotypeModel.PackageCurrentState()
        
        avg_epoch_test_loss = 0.0
        for i,data_batch in enumerate(testingDataLoader):
            if self.useCuda:
                data_batch=self._sendDataToGPU(data_batch)
            avg_epoch_test_loss+=sviFunction['svi'].evaluate_loss(*data_batch,minibatch_scale = (numTotalTestingSamples/data_batch[0].shape[0]))
            
        bestModelScore = avg_epoch_test_loss/(i+1)
        
        avg_epoch_train_loss = 0.0
        for i,data_batch in enumerate(trainingDataLoader):
            if self.useCuda:
                data_batch=self._sendDataToGPU(data_batch)
            avg_epoch_train_loss+=sviFunction['svi'].step(*data_batch,minibatch_scale = (numTotalTrainingSamples/data_batch[0].shape[0]))
        prev_train_loss=avg_epoch_train_loss/(i+1)
        
        elbo_window.append(prev_train_loss)
        
        for epoch in range(self.maxEpochs):
            torchTrainingData.shuffle_index()
            avg_epoch_train_loss = 0.0
            for i,data_batch in enumerate(trainingDataLoader):
                if self.useCuda:
                    data_batch=self._sendDataToGPU(data_batch)
                avg_epoch_train_loss+=sviFunction['svi'].step(*data_batch,minibatch_scale = (numTotalTrainingSamples/data_batch[0].shape[0]))
                    
            avg_epoch_train_loss=avg_epoch_train_loss/(i+1.0)
                
            
            
            if self.withCosineAnnealing:
                sviFunction['scheduler'].step(epoch=epoch)
            else:
                sviFunction['scheduler'].step(avg_epoch_train_loss)
                
                
            avg_epoch_test_loss = 0.0
            for i,data_batch in enumerate(testingDataLoader):
                if self.useCuda:
                    data_batch=self._sendDataToGPU(data_batch)
                avg_epoch_test_loss+=sviFunction['svi'].evaluate_loss(*data_batch,minibatch_scale = (numTotalTestingSamples/data_batch[0].shape[0]))
            avg_epoch_test_loss=avg_epoch_test_loss/(i+1)
            
            if avg_epoch_test_loss<bestModelScore:
                bestModelState = self.phenotypeModel.PackageCurrentState()
                bestModelScore = avg_epoch_test_loss
            
                
                
            #track overall convergence
            trainLoss+=[avg_epoch_train_loss]
            testLoss+=[avg_epoch_test_loss]
        
    
            
            error_window.append(rel_diff(avg_epoch_train_loss,prev_train_loss))
            elbo_window.append(avg_epoch_train_loss)
            
            avg_error = sum(error_window)/len(error_window)
            med_error = sorted(error_window)[int(0.5*len(error_window))]
            slope_error = self._returnELBOSlope(elbo_window)
            
            prev_train_loss=avg_epoch_train_loss
            errorVec+=[min([avg_error,med_error,slope_error])]
            
            if (avg_error < errorTol) or (med_error < errorTol) or (slope_error<errorTol):
                

                if verbose:
                    print("Optimization converged in %03d epochs; Current Loss (Train, Test): %.4f, %.4f; Error: %.4e"%(epoch+1,trainLoss[-1],testLoss[-1],min([avg_error,med_error,slope_error])))
                if logFile!=None:
                    with open(logFile, "a") as f:
                        f.write("Optimization converged in %03d epochs; Current Loss (Train, Test): %.4f, %.4f; Error: %.4e\n"%(epoch+1,trainLoss[-1],testLoss[-1],min([avg_error,med_error,slope_error])))
                break
            
            else:
                if verbose:
                    print("Completed %03d epochs; Current Loss (Train, Test): %.4f, %.4f; Error: %.4e"%(epoch+1,trainLoss[-1],testLoss[-1],min([avg_error,med_error,slope_error])))
                if logFile!=None:
                    with open(logFile, "a") as f:
                        f.write("Completed %03d epochs; Current Loss (Train, Test): %.4f, %.4f; Error: %.4e\n"%(epoch+1,trainLoss[-1],testLoss[-1],errorVec[-1][0],min([avg_error,med_error,slope_error])))
                
            
            
                
        if epoch == (self.maxEpochs-1):
            print('Warning: Optimization did not converge in allotted epochs')
        
        self.phenotypeModel.LoadPriorState(bestModelState)
        #set eval mode
        if self.useCuda:
            self.phenotypeModel.SwitchDevice('cpu')
        self.phenotypeModel.eval()
        return bestModelScore,trainLoss,testLoss,errorVec
