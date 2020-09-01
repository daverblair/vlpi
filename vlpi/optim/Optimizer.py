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
from pyro.optim import OneCycleLR
from torch.optim import AdamW
from collections import deque
import numpy as np
from scipy.stats import linregress
from pyro import poutine
from vlpi.data.ClinicalDataset import _TorchDatasetWrapper
from vlpi.utils.UtilityFunctions import rel_diff


class AnnealingScheduler:
    def _cosine_anneal(self,currentStepFrac):
        cos_vals = np.cos(np.pi * currentStepFrac) + 1.0
        return self.finalTemp + ((self.initTemp - self.finalTemp) / 2.0) * cos_vals

    def _linear_anneal(self,currentStepFrac):
        return (self.finalTemp - self.initTemp) * currentStepFrac + self.initTemp

    def __init__(self,initTemp,finalTemp,totalNumSteps,scheduleType):
        """
        Scheduler for KL-Anneling.

        Parameters
        ----------
        initTemp : float
            Initial temperature.
        finalTemp : float
            Final temperature.
        totalNumSteps : int
            Number of total steps for annealing
        scheduleType : str
            Type of scheduler. Must be 'linear' or 'cosine'.

        Returns
        -------
        None

        """

        assert initTemp>=0.0 and finalTemp >=0.0,"Annealing temperatures must be greater than 0.0"
        self.initTemp=initTemp
        self.finalTemp=finalTemp
        self.totalNumSteps=totalNumSteps
        if scheduleType=='linear':
            self._anneal_func=self._linear_anneal
        else:
            self._anneal_func=self._cosine_anneal

    def CurrentTemp(self,currentStep):
        frac = currentStep/self.totalNumSteps
        if frac>1.0:
            return self.finalTemp
        else:
            return self._anneal_func(frac)


class Optimizer:

    def _returnELBOSlope(self,elbo_deque):
        return abs(linregress(range(len(elbo_deque)),elbo_deque)[0]/np.mean(elbo_deque))

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




    def __init__(self,phenotypeModel,datasetSampler,optimizationParameters={'maxLearningRate': 0.05,'maxEpochs': 5000,'numParticles':10},computeConfiguration={'device':None,'numDataLoaders':0}, **kwargs):
        """

        This class implements two SGD optimizers, one of which uses the full training dataset (FullDatasetTrain) and the other uses mini-batches randomly sampled from the training dataset. To improve convergence rate and escape local minima, learning rate can be autommatically altered during inference. Note, given stochastic inference strategy coupled with possible re-starts, class tracks model perfomance and stores the best possible instance of the model obtained throughout the inference process. This also helps to avoid overfitting.

                phenotypeModel-->The phenotype model to optimize
                datasetSampler-->ClinicalDatasetSampler that holds clinical data and generates random samples

                maxLearningRate-->maximum learning rate used by SGD during inference.
                device-->compute device to use. By default, uses CPU unless GPU device specified, which can be string or integer

        Parameters
        ----------
        phenotypeModel : vlpi.model.VAE
            The phenotype model to optimize.
        datasetSampler : vlpi.data.ClinicalDatasetSampler
            ClinicalDatasetSampler that generates data
        optimizationParameters : dict
            Dictionary containing optimzation parameters. Default: {'maxLearningRate': 0.05,'maxEpochs': 5000,'numParticles':10}
        computeConfiguration : dict
            Dictionary containing the compute configuration (device and number of dataloaders. Defaults to using cpu: {'device':None,'numDataLoaders':0}
        **kwargs : See source code.
            These arguments adjust the parameters of the One-Cycle learning rate policy ('OneCycleParams'), Adam weight decay parameter ('AdamW_Weight_Decay'), and the KL-Annealing scheduler ('KLAnnealingParams'). See source code for details.

        Returns
        -------
        None

        """



        allKeywordArgs = list(kwargs.keys())

        self.phenotypeModel=phenotypeModel
        self.datasetSampler=datasetSampler

        #general optimization parameters
        self.maxEpochs=optimizationParameters['maxEpochs']
        self.maxLearningRate = optimizationParameters['maxLearningRate']
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




        if 'OneCycleParams' not in allKeywordArgs:
            self.OneCycleParams={'pctCycleIncrease':1.0,'initLRDivisionFactor':1.0,'finalLRDivisionFactor':1.0}

        else:
            self.OneCycleParams=kwargs['OneCycleParams']

        if 'AdamW_Weight_Decay' in allKeywordArgs:
            self.AdamW_Weight_Decay=kwargs['AdamW_Weight_Decay']
        else:
            self.AdamW_Weight_Decay=1e-4

        if 'KLAnnealingParams' not in allKeywordArgs:
            self.KLAnnealingParams = {'initialTemp':1.0,'maxTemp':1.0,'fractionalDuration':1.0,'schedule': 'cosine'}
        else:
            self.KLAnnealingParams=kwargs['KLAnnealingParams']
            assert set(self.KLAnnealingParams.keys())==set(['initialTemp','maxTemp','fractionalDuration','schedule']),"KL Annealing Parameters must be dictionary with the following keys: 'initialTemp','maxTemp','fractionalDuration','schedule'"





    def _contstructSVI(self,optimizationStrategy,total_steps):
        """
        Constructs a list of SVI functions use to train model. Takes optimizationStrategy as input, which must be in ['Full','PosteriorOnly','GuideOnly']
        """

        #generate random sample just to build the param_store
        test_data = self.datasetSampler.GenerateRandomTrainingSample(2)
        if self.useCuda:
                test_data=self._sendDataToGPU(test_data)
        self.phenotypeModel.guide(*test_data)
        param_store=pyro.get_param_store()

        if optimizationStrategy=='Full':
            _model=self.phenotypeModel.model
            _guide=self.phenotypeModel.guide

        else:
            if optimizationStrategy=='DecoderOnly':
                _model=self.phenotypeModel.model
                _guide=poutine.block(self.phenotypeModel.guide,hide=[x for x in param_store.get_all_param_names() if ('encoder' in x)])

            else:
                _model=poutine.block(self.phenotypeModel.model,hide=[x for x in param_store.get_all_param_names() if ('decoder' in x)])
                _guide=self.phenotypeModel.guide

        #initialize the SVI class
        AdamWOptimArgs = {'weight_decay':self.AdamW_Weight_Decay}

        scheduler = OneCycleLR({'optimizer': AdamW, 'optim_args': AdamWOptimArgs,'max_lr':self.maxLearningRate,'total_steps':total_steps,'pct_start':self.OneCycleParams['pctCycleIncrease'],'div_factor':self.OneCycleParams['initLRDivisionFactor'],'final_div_factor':self.OneCycleParams['finalLRDivisionFactor']})
        return {'svi':SVI(_model,_guide,scheduler, loss=Trace_ELBO(num_particles=self.numParticles)),'scheduler':scheduler}


    def FullDatasetTrain(self, errorTol:float = 1e-3,verbose:bool=True,logFile=None,optimizationStrategy='Full',errorComputationWindow=None):
        """

        Trains the VAE model using the full dataset. Not recommended.

        Parameters
        ----------
        errorTol : float
            Error tolerance in validation data ELBO for convergence.
        verbose : bool
            Whether to print updates regarding fitting.
        logFile : str
            Path to logfile which can store progress
        optimizationStrategy : str
            Allows user to update only certain parts of model. Useful for debugging. Must be one of: 'Full','EncoderOnly','DecoderOnly'
        errorComputationWindow : float
            Sliding window for computing error tolerance. Default appears to work well.

        Returns
        -------
        tuple
            (bestModelScore, vector of ELBO evaluations on training data,vector of ELBO evaluations on testing data, vector of all errors computed during fitting)

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



        assert optimizationStrategy in ['Full','EncoderOnly','DecoderOnly'],"Available Optimization strategies: ['Full','EncoderOnly','DecoderOnly']"
        sviFunction = self._contstructSVI(optimizationStrategy,self.maxEpochs)
        annealer =AnnealingScheduler(self.KLAnnealingParams['initialTemp'],self.KLAnnealingParams['maxTemp'],int(self.maxEpochs*self.KLAnnealingParams['fractionalDuration']),self.KLAnnealingParams['schedule'])
        paramUpdateNum = 0


        # initialize model and parameter storage vectors. Note, parameter storage vectors are stored in a list with each element corresponding to one SVI function
        bestModelState= self.phenotypeModel.PackageCurrentState()
        bestModelScore = sviFunction['svi'].evaluate_loss(*testData,annealing_factor=annealer.CurrentTemp(paramUpdateNum))
        prev_train_loss = sviFunction['svi'].evaluate_loss(*trainData,annealing_factor=self.KLAnnealingParams['maxTemp'])
        elbo_window.append(prev_train_loss)

        for epoch in range(self.maxEpochs):
            self._localShuffle(trainData)

            # take step on training data

            paramUpdateNum+=1
            currrent_train_loss=sviFunction['svi'].step(*trainData,annealing_factor=annealer.CurrentTemp(paramUpdateNum))

            #adjust learning rate per scheduler
            sviFunction['scheduler'].step()


            #no point in shuffling testing data, all observations are independent
            #also, all svi functions should return identical loss, so we arbitrarily choose the latter for loss
            current_test_loss=sviFunction['svi'].evaluate_loss(*testData,annealing_factor=self.KLAnnealingParams['maxTemp'])

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

        Trains the VAE model using the mini-batches. This is the recommended method.

        Parameters
        ----------
        batch_size : int
            Number of samples in each mini-batch
        errorTol : float
            Error tolerance in validation data ELBO for convergence.
        verbose : bool
            Whether to print updates regarding fitting.
        logFile : str
            Path to logfile which can store progress
        optimizationStrategy : str
            Allows user to update only certain parts of model. Useful for debugging. Must be one of: 'Full','EncoderOnly','DecoderOnly'
        errorComputationWindow : float
            Sliding window for computing error tolerance. Default appears to work well.

        Returns
        -------
        tuple
            (bestModelScore, vector of ELBO evaluations on training data,vector of ELBO evaluations on testing data, vector of all errors computed during fitting)

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



        assert optimizationStrategy in ['Full','EncoderOnly','DecoderOnly'],"Available Optimization strategies: ['Full','EncoderOnly','DecoderOnly']"
        sviFunction = self._contstructSVI(optimizationStrategy,len(torchTrainingData)*self.maxEpochs)
        annealer=AnnealingScheduler(self.KLAnnealingParams['initialTemp'],self.KLAnnealingParams['maxTemp'],int(len(torchTrainingData)*(self.maxEpochs*self.KLAnnealingParams['fractionalDuration'])),self.KLAnnealingParams['schedule'])
        paramUpdateNum=0


        #initialize model states and scores
        bestModelState= self.phenotypeModel.PackageCurrentState()

        #note, batch_size handled at the level of _TorchDatasetWrapper, not loader itself.
        trainingDataLoader = data.DataLoader(torchTrainingData,batch_size=1,shuffle=False,sampler=None,batch_sampler=None,num_workers=self.num_dataloaders,collate_fn=lambda x:x[0],pin_memory=self.useCuda)
        testingDataLoader = data.DataLoader(torchTestingData,batch_size=1,shuffle=False,sampler=None,batch_sampler=None,num_workers=self.num_dataloaders,collate_fn=lambda x:x[0],pin_memory=self.useCuda)

        avg_epoch_train_loss = 0.0
        for i,data_batch in enumerate(trainingDataLoader):
            if self.useCuda:
                data_batch=self._sendDataToGPU(data_batch)

            avg_epoch_train_loss+=sviFunction['svi'].evaluate_loss(*data_batch,minibatch_scale = (numTotalTrainingSamples/data_batch[0].shape[0]),annealing_factor=annealer.CurrentTemp(paramUpdateNum+1))
        prev_train_loss=avg_epoch_train_loss/(i+1)

        elbo_window.append(prev_train_loss)

        avg_epoch_test_loss = 0.0

        for i,data_batch in enumerate(testingDataLoader):
            if self.useCuda:
                data_batch=self._sendDataToGPU(data_batch)
            avg_epoch_test_loss+=sviFunction['svi'].evaluate_loss(*data_batch,minibatch_scale = (numTotalTestingSamples/data_batch[0].shape[0]),annealing_factor=self.KLAnnealingParams['maxTemp'])
        bestModelScore = avg_epoch_test_loss/(i+1)


        for epoch in range(self.maxEpochs):
            torchTrainingData.shuffle_index()
            avg_epoch_train_loss = 0.0
            for i,data_batch in enumerate(trainingDataLoader):
                paramUpdateNum+=1
                if self.useCuda:
                    data_batch=self._sendDataToGPU(data_batch)
                avg_epoch_train_loss+=sviFunction['svi'].step(*data_batch,minibatch_scale = (numTotalTrainingSamples/data_batch[0].shape[0]),annealing_factor=annealer.CurrentTemp(paramUpdateNum))
                sviFunction['scheduler'].step()


            avg_epoch_train_loss=avg_epoch_train_loss/(i+1.0)

            avg_epoch_test_loss = 0.0
            for i,data_batch in enumerate(testingDataLoader):
                if self.useCuda:
                    data_batch=self._sendDataToGPU(data_batch)
                avg_epoch_test_loss+=sviFunction['svi'].evaluate_loss(*data_batch,minibatch_scale = (numTotalTestingSamples/data_batch[0].shape[0]),annealing_factor=self.KLAnnealingParams['maxTemp'])
            avg_epoch_test_loss=avg_epoch_test_loss/(i+1)

            if np.isnan(avg_epoch_test_loss) or np.isnan(avg_epoch_train_loss):
                print("Warning: NaN detected during inference. Model unlikely to be fully optimized!")
                break

            elif avg_epoch_test_loss<bestModelScore:
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
