#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:18:43 2019

@author: davidblair
"""

import torch
import pyro
import pyro.distributions as dist
from sklearn.utils import shuffle
import numpy as np
from pyro.infer import TraceEnum_ELBO,Trace_ELBO
from vlpi.Optimizers import Optimizers
from vlpi.vLPIDiscrete import DiscreteModel
from vlpi.ClinicalDataset import ClinicalDatasetSampler,ClinicalDataset
from vlpi.UnsupervisedMethods import PheRS,JaccardSimilarity,LinearEmbedding
from vlpi.vLPIContinuous import ContinuousModel
from vlpi.vLPIDiscriminative import DiscriminativeModel
from vlpi.utils import infer_liability_CI
from sklearn.feature_selection import SelectFdr,chi2
from vlpi.utils import fisher_exact


class vLPI:
    
    def _samplerCheck(self,sampler):
        assert sampler.isConditioned==True,"Dataset provided in sampler is not conditioned on any dx. This is required when providing a anchor dx"
        assert self.useAvailableAnchorDx in sampler.conditionSamplingOnDx, "Anchor dx {} is not in in list of conditioned diagnoses for dataset sampler".format(self.useAvailableAnchorDx)
            
    
    def _parseArgsDiscriminative(self,**kwargs):
        allKeywordArgs = list(kwargs.keys()) 
        assert 'DiscreteModelParameters' not in allKeywordArgs, "Discrete model parameters specified for a Discriminative Model. Please specify 'DiscriminativeModelParameters'"
            
        assert 'ContinuousModelParameters' not in allKeywordArgs, "Continuous model parameters specified for a Discriminative Model. Please specify 'DiscriminativeModelParameters'"
        
        if 'DiscriminativeModelParameters' not in allKeywordArgs:
            modelParameters={}
            modelParameters['linkFunction']='Probit'
            modelParameters['coupleCovariates']=False
            
        else:
            assert isinstance(kwargs['DiscriminativeModelParameters'],dict),"Expect 'DiscriminativeModelParameters' to be a dictionary of paramters"
            assert set(kwargs['DiscriminativeModelParameters'].keys())==set(['linkFunction','coupleCovariates']),"""DiscriminativeModelParameters must contain the following key-value pairs:
                KEY                      VALUE
                linkFunction             Link function for Discriminative model. Default is 'Probit'. Options are 'Probit' or 'Logit'
                coupleCovariates         Boolean value indictating whether or not to couple the covariates into the non-linear function used to map predictors to response. If False, they will be incorporated as linear predictors. Default is True.
                
                """
            modelParameters={}
            for key,value in kwargs['DiscriminativeModelParameters'].items():
                modelParameters[key]=value
        return modelParameters
            
        
    
    def _parseArgsContinuous(self,**kwargs):
        
        allKeywordArgs = list(kwargs.keys()) 
        
        assert 'DiscreteModelParameters' not in allKeywordArgs, "Discrete model parameters specified for a Continuous Model. Please specify 'ContinuousModelParameters'"
            
        assert 'DiscriminativeModelParameters' not in allKeywordArgs, "Discriminative model parameters specified for a Continuous Model. Please specify 'ContinuousModelParameters'"
        
        if 'ContinuousModelParameters' not in allKeywordArgs:
            
            modelParameters={}
            modelParameters['nLatentDim']=1
            modelParameters['anchorDxPriors']={'anchorDxNoise':[1.0,1.0],'latentDimToAnchorDxMap':1.0,'prevalence':[-3.0,3.0]}
            modelParameters['latentPhenotypePriors']={'element_wise_precision':[1.0,1.0]}
            modelParameters['covariatePriors']={'intercept':[0.0,5.0],'cov_scale':3.0}
        else:
            assert isinstance(kwargs['ContinuousModelParameters'],dict),"Expect 'ContinuousModelParameters' to be a dictionary of paramters"
            assert set(kwargs['ContinuousModelParameters'].keys())==set(['nLatentDim','anchorDxPriors','latentPhenotypePriors','covariatePriors']),"""ContinuousModelParameters must contain the following key-value pairs:
                KEY                      VALUE
                nLatentDim               Integer value indicating number of latent dimensions to include in the model. Default is 1. 
                anchorDxPriors           Dictionary with two entries: {'anchorDxNoise':[conc,rate],'prevalence':[low,high]}, where [conc,rate] denote gamma prior parameters over gaussian distributed noise, and [low,high] represent low and high ends of a 95% CI that represents the prior over latent phenotype prevalence (each value must be between 0.0 and 1.0). Can be VERY broad.
                latentPhenotypePriors    Dictionary with one entry: {'element_wise_precision':[conc,rate]}. [conc,rate] represent gamma prior parameters specifying distribution over the precision of exponentially-distributed latent phenotype effects. 
                
                covariatePriors:    Dictionary with two entries: {'intercept':[mean,scale],'cov_scale':scale}. Mean, scale represent mean and scale parameters of a gaussian prior over observed trait incidence. In cases in which only scale is provided, mean is assumed to be zero.
                
                """
            modelParameters={}
            for key,value in kwargs['ContinuousModelParameters'].items():
                modelParameters[key]=value
        return modelParameters
    
    def _parseArgsDiscrete(self,**kwargs):
        
        allKeywordArgs = list(kwargs.keys()) 
        
        assert 'ContinuousModelParameters' not in allKeywordArgs, "Continous model parameters specified for a Discrete Model. Please specify 'DiscreteModelParameters'"
            
        assert 'DiscriminativeModelParameters' not in allKeywordArgs, "Discriminative model parameters specified for a Discrete Model. Please specify 'DiscreteModelParameters'"
        
        if 'DiscreteModelParameters' not in allKeywordArgs:
            modelParameters={}
            modelParameters['nLatentDim']=1
            modelParameters['anchorDxPriors']={'anchorDxNoise':[1.0,1.0]}
            modelParameters['latentPhenotypePriors']={'element_wise_precision':[1.0,1.0],'prevalence':[-3.0,3.0]}
            modelParameters['covariatePriors']={'intercept':[0.0,5.0],'cov_scale':3.0}
        else:
            assert isinstance(kwargs['DiscreteModelParameters'],dict),"Expect 'DiscreteModelParameters' to be a dictionary or paramters"
            assert set(kwargs['DiscreteModelParameters'].keys())==set(['anchorDxPriors','latentPhenotypePriors','covariatePriors']),"""DiscreteModelPriorParameters must contain the following key-value pairs:
                KEY                      VALUE
                anchorDxPriors            Dictionary with one entry: {'anchorDxNoise':[conc,rate]}, where conc,rate denote gamma prior parameters over gaussian distributed noise.
                latentPhenotypePriors    Dictionary with two entries: {'element_wise_precision':[conc,rate],'prevalence':[low,high]}. conc,rate represent gamma prior parameters specifying distribution over the precision of exponentially-distributed latent phenotype effects. [low,high] represent low and high ends of a 95% CI that represents the prior over latent phenotype prevalence (each value must be between 0.0 and 1.0). Can be VERY broad.
                
                covariatePriors:    Dictionary with two entries: {'intercept':[mean,scale],'cov_scale':scale}. Mean, scale represent mean and scale parameters of a gaussian prior over observed trait incidence. In cases in which only scale is provided, mean is assumed to be zero.
                
                """
            modelParameters={}
            modelParameters['nLatentDim']=1
            for key,value in kwargs['DiscreteModelParameters'].items():
                modelParameters[key]=value
        return modelParameters
        
    
    def __init__(self,modelType,useAvailableAnchorDx=None,**kwargs):
        """

        vLPI a statistical model that maps a latent phenotypic space to some vector of observed (binary) phenotypes through a function f(z), where z is the latent phenotypic space of interest. The latent phenotypic space Z can be binary or continouus, and in the latter case, may be multi-dimenisional.
        
        Inference of the model is conducted using a variational approximation of the model marginal model likelihood, which is estimated using gradient descent. To allow inference to scale to millions of patients and enable subsequent portability, latent phenotype inference is amoritized using a non-linear neural network.
        
        
        To improve inferential accuracy,the model can also use information provided in the form of labels marking the extreme end of the latent phenotypic spectrum, which we call anchor diagnoses. This can be thought of as the noisy output of some measurement of the latent phenotypic state. In such settings, the model corresponds to a form of "noisy" supervised inference.
        
        Formal mathematical definition of the model can be found in Blair et al ***.
        
        Arguments:
            
            modelType: 'Discrete','Continuous', or 'Discriminative'
                Discrete-->Latent phenotypic space is binary (has disease, no disease)
                Continuous-->Latent phenotypic space is follows a (potentially multidimensional) standard gaussian distriibution
                Discriminative-->anchor diagnoses are treated as fixed labels, and a linear/non-linear function with probit link is used to predict said labels.
                
            useAvailableAnchorDx: Variable that indicates whether a anchor diagnosis is available in the clinical dataset provided for inference. If true, the ClinicalDatasetSampler must reference a version of the ClinicalDataset that has been conditioned on the anchor dx. See ClinicalDataset class for more information. 
            
        **kwargs: keyword arguments that can be passed to an arbitrary model. They consist of the following:
                        
            dropLinearCovariateColumn: boolean value indicated whether to drop one category from each covariate included into the model. Aids in identifiability. Defaults to True.
            
            neuralNetworkHyperparameters: speficies hyperparameters of the encoder/decoder neural networks. Default is a 2-layer MLP with 128 hidden nodes per layer. Dropout rate 0.2 with Batch-Normalization enabled. 
    
        """
        assert modelType in ['Discrete','Continuous','Discriminative'], 'Only Discrete, Continuous and Discriminative Models supported.'
        self.modelType=modelType
        self.useAvailableAnchorDx=useAvailableAnchorDx
        if useAvailableAnchorDx is not None:
            assert isinstance(useAvailableAnchorDx,str), "useAvailableAnchorDx expects a string indicating an anchor dx."
        
        
        if self.modelType == 'Discriminative':
            assert self.useAvailableAnchorDx is not None,"Discriminative modeling requires a anchor dx (ie is supervised)"
        self.allmodel_kwargs = kwargs
        self.currentModel = None
        
        
    
    def SimulateData(self, numSamples,numAssociatedTraits,numMaskedTraits = 0, numNoiseTraits = 0,numCategoricalCovariates=[],**kwargs):
        """
        Enables arbitrary simulation of observed trait data from a variety of underlying models. The simulated need not match the main model defined during construction of the vLPI class, but will do so if provided no additional information. 
        
        Arguments:
            numSamples-->number of samples to simulate
            numAssociatedTraits-->number of traits associated with the latent phenotype of interest
            numMaskedTraits-->number of traits to mask (ie remove) in the returned dataset
            numNoiseTraits-->number of uncorrelated traits to include (annotation errors)
            numCategoricalCovariates-->list containing number of covariates per category to include into the model. Default is none (empty list).
        kwargs:
            Allow extensive customization of simulation. See source code comments and assertions for details.
        
        """
        pyro.clear_param_store()
        
        
        allKeywordArgs = list(kwargs.keys()) 
        
        if 'simModelType' not in allKeywordArgs:
            assert self.modelType!='Discriminative',"Discriminative model not simulated directly. Please specify simModelType='Discrete' or 'Continuous' in kwargs"
            simModelType = self.modelType
        else:
            simModelType = kwargs['simModelType']
            assert simModelType in ['Discrete','Continuous'],'Only discrete and continuous models available for simulation.'
            
            
        if simModelType == 'Discrete':
            modelParameters=self._parseArgsDiscrete(**kwargs)
            simModel = DiscreteModel(numAssociatedTraits,numCategoricalCovariates,self.useAvailableAnchorDx is not None,'Linear_Monotonic',anchorDxPriors=modelParameters['anchorDxPriors'],latentPhenotypePriors=modelParameters['latentPhenotypePriors'],covariatePriors=modelParameters['covariatePriors'],**self.allmodel_kwargs)
            
            
                
        else:
            modelParameters = self._parseArgsContinuous(**kwargs)
            simModel = ContinuousModel(numAssociatedTraits,numCategoricalCovariates,self.useAvailableAnchorDx is not None,modelParameters['nLatentDim'],'Linear_Monotonic',anchorDxPriors=modelParameters['anchorDxPriors'],latentPhenotypePriors=modelParameters['latentPhenotypePriors'],covariatePriors=modelParameters['covariatePriors'],**self.allmodel_kwargs)
                  
        _baselineSimOutput = simModel.SimulateData(numSamples)
        
        
        if numMaskedTraits>0:
            unmasked = shuffle(np.arange(numAssociatedTraits))[numMaskedTraits:]
            _baselineSimOutput['incidence_data']=_baselineSimOutput['incidence_data'][:,unmasked]
            _baselineSimOutput['model_params']['intercepts']=_baselineSimOutput['model_params']['intercepts'][unmasked]
            _baselineSimOutput['model_params']['latentPhenotypeEffects']=_baselineSimOutput['model_params']['latentPhenotypeEffects'][:,unmasked]
            if len(numCategoricalCovariates)>0:
                _baselineSimOutput['model_params']['covEffects']=_baselineSimOutput['model_params']['covEffects'][:,unmasked]
                
        if numNoiseTraits>0:
            new_order = shuffle(np.arange((numAssociatedTraits-numMaskedTraits)+numNoiseTraits))
            noisePrevalence = dist.Normal(simModel.priorParamDict['intercepts']['mean'],simModel.priorParamDict['intercepts']['scale']).sample([numNoiseTraits])
            
            noiseIncidenceValues = dist.Bernoulli(dist.Normal(0.0,1.0).cdf(noisePrevalence)).sample([numSamples])
            
            _baselineSimOutput['incidence_data']=torch.cat((_baselineSimOutput['incidence_data'],noiseIncidenceValues),dim=-1)[:,new_order]
            _baselineSimOutput['model_params']['intercepts']=torch.cat((_baselineSimOutput['model_params']['intercepts'],noisePrevalence),dim=-1)[new_order]
            _baselineSimOutput['model_params']['latentPhenotypeEffects']=torch.cat((_baselineSimOutput['model_params']['latentPhenotypeEffects'],torch.zeros(modelParameters['nLatentDim'],numNoiseTraits,dtype=torch.float32)),dim=-1)[:,new_order]
            
            if len(numCategoricalCovariates)>0:
                _baselineSimOutput['model_params']['covEffects']=torch.cat((_baselineSimOutput['model_params']['covEffects'],torch.zeros(_baselineSimOutput['model_params']['covEffects'].shape[0],numNoiseTraits,dtype=torch.float32)),dim=-1)[:,new_order]
        pyro.clear_param_store()
        
        if simModelType=='Continuous':
            
            if self.useAvailableAnchorDx is None:
                lp_prev = dist.Normal(modelParameters['anchorDxPriors']['prevalence'][0],modelParameters['anchorDxPriors']['prevalence'][1]).sample()
                _baselineSimOutput['model_params']['latentPhenotypePrevalence']=lp_prev
                if modelParameters['nLatentDim']>1:
                    lp_map = dist.Dirichlet((modelParameters['anchorDxPriors']['latentDimToAnchorDxMap']/float(modelParameters['nLatentDim']))*torch.ones(modelParameters['nLatentDim'])).sample()
                    _baselineSimOutput['model_params']['latentDimToAnchorMap']=lp_map
                
            
        return _baselineSimOutput
            

#    def ElasticNet(self,datasetSampler,verbose=True,**kwargs):
#        """
#        Thin wrapper around elastic net regression in sklearn library.
#        
#        Keyword Arguments:
#        
#            errorTol: errorTolerance for general model fitting. Default is 1e-4
#        
#            maxEpochs: Maximum number of epochs for model fitting. Default is 1000
#            
#            penalty_param: penalty parameter for fitting, default is 3.0. See sklearn for details
#            
#            l1_ratio: ratio of penalty attributed to l1 regularization. See sklearn for details.
#        """
#        
#        allKeywordArgs = list(kwargs.keys()) 
#        if 'errorTol' in allKeywordArgs:
#            errorTol=kwargs['errorTol']
#        else:
#            errorTol=1e-4
#        
#
#        if 'maxEpochs' in allKeywordArgs:
#            maxEpochs = kwargs['maxEpochs']
#        else:
#            maxEpochs = 1000
#            
#        if 'penalty_param' in allKeywordArgs:
#            penalty_param=kwargs['penalty_param']
#        else:
#            penalty_param=3.0
#         
#        if 'l1_ratio' in allKeywordArgs:
#            l1_ratio=kwargs['l1_ratio']
#        else:
#            l1_ratio=0.5
#        
#        
#        
#        previousArrayType = datasetSampler.returnArrays
#        if datasetSampler.returnArrays!='Sparse':
#            datasetSampler.ChangeArrayType('Sparse')
#        
#        trainData = datasetSampler.ReturnFullTrainingDataset()
#        X_data_train = datasetSampler.CollapseDataArrays(trainData[0],trainData[1],drop_column=True)
#
#        
#        
#        
#        anchor_prev = trainData[2].sum()/trainData[2].shape[0]
#        
#        self.currentModel.intercept_ = np.array([np.log(anchor_prev)-np.log(1.0-anchor_prev)])
#        self.currentModel.coef_ = np.zeros((1,X_data_train.shape[1]))
#        
#        self.currentModel.C = penalty_param
#        self.currentModel.l1_ratio=l1_ratio
#        
#        self.currentModel.fit(X_data_train,trainData[2].toarray().ravel())
#        
#        testData = datasetSampler.ReturnFullTestingDataset()
#        X_data_test = datasetSampler.CollapseDataArrays(testData[0],testData[1],drop_column=True)
#        
#        log_loss_train=log_loss(trainData[2],self.currentModel.predict_proba(X_data_train))
#        log_loss_test=log_loss(testData[2],self.currentModel.predict_proba(X_data_test))
#        
#        if previousArrayType!='Sparse':
#            datasetSampler.ChangeArrayType(previousArrayType)
#        return log_loss_train,log_loss_test
        
    
    def FitModel(self,datasetSampler,batch_size=0,verbose=True,**kwargs):
        """
        datasetSampler: Class of ClinicalDatasetSampler, which provides access to data stored in the form of an instantiation ClinicalDataset (essentially a sparse array).
        
        batch_size: size of mini-batches for stochastic inference. Default is 0, which indicates that the full dataset will be used.
        verbose: indicates whether or not to print out inference progress in real time
        
        Keyword Arguments:
            alreadyInitialized: boolean, indicates whether the model has been pre-initialized (by copying from previous state). This way initialization procedure can be ignored. Can also use to skip initialization procedure on un-initialized model, at which point it is initialize witht the prior. 
            
            mappingFunction: Function used to map latent phenotype state (or observed phenotypes in the case of discriminative models. Default is Linear_Monotonic, which enforces a positive correlation structure among all observed phenotypes. Available function ['Linear','Linear_Monotonic'] for latent phenotype models, ['Linear','Linear_Monotonic', 'Nonlinear', 'Nonlinear_Monotonic'] for discriminative model.
            
            learningRate: initial learning rate for the stochastic gradient descent. Default is 0.005.
            
            withCosineAnealing: boolean, indicates whether to use cosine annealing with warm restarts to adjust learning rate during inference. Default is False.
            
            initializationErrorTol: error tolerance for initialization procedure. Default is 0.01.
            
            errorTol: errorTolerance for general model fitting. Default is 1e-4
        
            maxEpochs: Maximum number of epochs for model fitting. Default is 1000
            
            computeDevice: Device to use for model fitting. Default is None, which specifies cpu. Use integer (corresponding to device number) to specify gpu.
            
            numDataLoaders: number of dataloaders used for gpu computing, as model fitting becomes I/O bound if using gpu. If not gpu enabled, must be 0. 
        """
        
        ######### Parse Keyword Arguments #########
        allKeywordArgs = list(kwargs.keys()) 
        
        #allows user to keep the param store the same if re-starting optimization, for example, after loading from disk
        if 'alreadyInitialized' in allKeywordArgs:
            if kwargs['alreadyInitialized']==False:
                pyro.clear_param_store()
            alreadyInitialized=kwargs['alreadyInitialized']
        else:
            alreadyInitialized=False
            pyro.clear_param_store()

            
        if 'mappingFunction' in allKeywordArgs:
                mappingFunction=kwargs['mappingFunction']
        else:
            mappingFunction='Linear_Monotonic'
            
        
        if self.modelType!='Discriminative':
            assert mappingFunction in ['Linear','Linear_Monotonic'], "Only Linear and Linear_Monotonic mapping functions current supported for latent variable models."
        if 'learningRate' in allKeywordArgs:
            learningRate=kwargs['learningRate']
        else:
            learningRate=0.005
        
        if 'numParticles' in allKeywordArgs:
            numParticles=kwargs['numParticles']
        else:
            numParticles=10
            
        
        if 'withCosineAnnealing' in allKeywordArgs:
            withCosineAnealing=kwargs['withCosineAnnealing']
        else:
            withCosineAnealing=False
        
        cosineAnnealingDict = {'withCosineAnnealing':withCosineAnealing,'initialRestartInterval':10,'intervalMultiplier':2}
            
        if 'initializationErrorTol' in allKeywordArgs:
            initializationErrorTol=kwargs['initializationErrorTol']
        else:
            initializationErrorTol = 1e-2
            
        if 'maxInitializationEpochs' in allKeywordArgs:
            maxInitializationEpochs=kwargs['maxInitializationEpochs']
        else:
            maxInitializationEpochs=20
        
        if 'errorTol' in allKeywordArgs:
            errorTol=kwargs['errorTol']
        else:
            errorTol=1e-4
        

        if 'maxEpochs' in allKeywordArgs:
            maxEpochs = kwargs['maxEpochs']
        else:
            maxEpochs = 1000
            
        if 'computeDevice' in allKeywordArgs:
            computeDevice=kwargs['computeDevice']
        else:
            computeDevice=None
        
        if 'numDataLoaders' in allKeywordArgs:
            numDataLoaders=kwargs['numDataLoaders']
            if computeDevice in [None,'cpu']:
                assert numDataLoaders==0,"Specifying number of dataloaders other than 0 only relevant when using GPU computing"
        else:
            numDataLoaders=0

        
        
        if self.useAvailableAnchorDx is not None:
            self._samplerCheck(datasetSampler)
        
        #############################################
        
        numObsTraits = datasetSampler.currentClinicalDataset.numDxCodes
        numCatList=[len(x) for x in datasetSampler.currentClinicalDataset.catCovConversionDicts.values()]
        
        #############################################
        
        
        ####### Now instantiate the model, if it doesn't exist already #######
        if self.currentModel is None:
        
        
            if self.modelType=='Discrete':
                modelParameters=self._parseArgsDiscrete(**kwargs)
                self.currentModel = DiscreteModel(numObsTraits, numCatList,self.useAvailableAnchorDx is not None,mappingFunction,anchorDxPriors=modelParameters['anchorDxPriors'],latentPhenotypePriors=modelParameters['latentPhenotypePriors'],covariatePriors=modelParameters['covariatePriors'],**self.allmodel_kwargs)
                    
            elif self.modelType=='Continuous':
                modelParameters=self._parseArgsContinuous(**kwargs)
                self.currentModel = ContinuousModel(numObsTraits, numCatList,self.useAvailableAnchorDx is not None,modelParameters['nLatentDim'],mappingFunction,anchorDxPriors=modelParameters['anchorDxPriors'],latentPhenotypePriors=modelParameters['latentPhenotypePriors'],covariatePriors=modelParameters['covariatePriors'],**self.allmodel_kwargs)
                
            else:
                modelParameters = self._parseArgsDiscriminative(**kwargs)
                self.currentModel = DiscriminativeModel(numObsTraits, numCatList,mappingFunction,linkFunction=modelParameters['linkFunction'],coupleCovariates=modelParameters['coupleCovariates'],**self.allmodel_kwargs)
        
        
        ###### Set the loss function used for fitting
        if self.modelType=='Discrete':
            lossFunc = TraceEnum_ELBO(max_plate_nesting=1,num_particles=numParticles)
        else:
            lossFunc = Trace_ELBO(num_particles=numParticles)
            
        ###### For generative models, overall fitting is improved by initialization, which uses a two-stage procedure.
        if self.modelType!='Discriminative' and alreadyInitialized==False:
            

            try:
                #check if model is continuous, if so, initialize with same number of dimensions
                trainScores,testScores = self.ComputeEmbeddings(datasetSampler,num_components=self.currentModel.nLatentDim)
            except AttributeError:
                trainScores,testScores = self.ComputeEmbeddings(datasetSampler,num_components=1)
                
            scoreVector = np.concatenate([trainScores,testScores])
                            
            if datasetSampler.isConditioned:
                allSampleIndex = np.concatenate([np.concatenate(datasetSampler.trainingDataIndex),np.concatenate(datasetSampler.testDataIndex)])
            else:
                allSampleIndex = np.concatenate([datasetSampler.trainingDataIndex,datasetSampler.testDataIndex])
            
            datasetSampler.AddScoresToDataset(scoreVector,allSampleIndex)
            
#            
#            #initialize encoder bu matching to a rank based encoder
            optimizer=Optimizers(self.currentModel,datasetSampler,Trace_ELBO(num_particles=numParticles),optimizationParameters={'initialLearningRate': learningRate,'maxEpochs': maxInitializationEpochs},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},cosineAnnealing=cosineAnnealingDict)
            
            if batch_size > 0:
                output=optimizer.BatchTrain(batch_size,errorTol=initializationErrorTol,optimizationStrategy='ScoreBased',verbose=verbose)
            else:
                output=optimizer.FullDatasetTrain(errorTol=initializationErrorTol,optimizationStrategy='ScoreBased',verbose=verbose)
            datasetSampler.RemoveScoresFromDataset()
            for i in range(10):
            
                optimizer=Optimizers(self.currentModel,datasetSampler,lossFunc,optimizationParameters={'initialLearningRate': learningRate,'maxEpochs': maxInitializationEpochs},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},cosineAnnealing=cosineAnnealingDict)
            
            
                if batch_size > 0:
                    output=optimizer.BatchTrain(batch_size,errorTol=initializationErrorTol,optimizationStrategy='PosteriorOnly',verbose=verbose)
                else:
                    output=optimizer.FullDatasetTrain(errorTol=initializationErrorTol,optimizationStrategy='PosteriorOnly',verbose=verbose)
                    
                optimizer=Optimizers(self.currentModel,datasetSampler,lossFunc,optimizationParameters={'initialLearningRate': learningRate,'maxEpochs': maxInitializationEpochs},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},cosineAnnealing=cosineAnnealingDict)
                if batch_size > 0:
                    output=optimizer.BatchTrain(batch_size,errorTol=initializationErrorTol,optimizationStrategy='EncoderOnly',verbose=verbose)
                else:
                    output=optimizer.FullDatasetTrain(errorTol=initializationErrorTol,optimizationStrategy='EncoderOnly',verbose=verbose)
                

        
        cosineAnnealingDict['withCosineAnnealing']=False
        optimizer=Optimizers(self.currentModel,datasetSampler,lossFunc,optimizationParameters={'initialLearningRate': learningRate,'maxEpochs': maxEpochs},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},cosineAnnealing=cosineAnnealingDict)

        if batch_size > 0:
            output=optimizer.BatchTrain(batch_size,errorTol=errorTol,optimizationStrategy='Full',verbose=verbose)
        else:
            output=optimizer.FullDatasetTrain(errorTol=errorTol,optimizationStrategy='Full',verbose=verbose)
##            
        return output
    
    def SelectComorbidTraits(self,datasetSampler,FDR,modifyDataset=False,useChi2=True):
        """
        Selects comorbid traits from the dataset at an false discovery rate of FDR.
        
        datasetSampler: sampler class
        FDR: false discovery rate
        modifyDataset: indicates whether to modify datset such that only comorbid terms are included in further analyses. Does not modify on disk.
        useChi2: whether chi2 (default) should be used for comorbidity. Alternative is fisher's exact test, which is slower
        
        """
        
        
        assert self.useAvailableAnchorDx is not None,"Must provide anchor dx to compute comorbid traits"
        self._samplerCheck(datasetSampler)
        previousArrayType = datasetSampler.returnArrays
        if datasetSampler.returnArrays!='Sparse':
            datasetSampler.ChangeArrayType('Sparse')
            
        sparseTrainingData=datasetSampler.ReturnFullTrainingDataset(randomize=False)
        dataMatrix=sparseTrainingData[0]
        incidenceVec =sparseTrainingData[2]
        
        if useChi2==False:
            fdr=SelectFdr(fisher_exact, alpha=FDR)
        else:
            fdr=SelectFdr(chi2, alpha=FDR)
            
            
        fdr_fit = fdr.fit(dataMatrix,incidenceVec.toarray())
        discIndx=np.where(fdr_fit.get_support()==True)[0]
        
        if modifyDataset:
            datasetSampler.currentClinicalDataset.IncludeOnly([datasetSampler.currentClinicalDataset.dataIndexToDxCodeMap[x] for x in discIndx])
        
        if previousArrayType!='Sparse':
            datasetSampler.ChangeArrayType(previousArrayType)
        
        return discIndx, fdr_fit.scores_[discIndx],fdr_fit.pvalues_[discIndx]
    
    
    def ComputePheRS(self,datasetSampler,includeCovariates=False):
        previousArrayType = datasetSampler.returnArrays
        if datasetSampler.returnArrays!='Sparse':
            datasetSampler.ChangeArrayType('Sparse')
        sparseTrainingData=datasetSampler.ReturnFullTrainingDataset(randomize=False)
        sparseTestingData=datasetSampler.ReturnFullTestingDataset(randomize=False)
        if includeCovariates:
            train_cov_matrix = datasetSampler.CollapseDataArrays(cov_vecs=sparseTrainingData[1])
            test_cov_matrix = datasetSampler.CollapseDataArrays(cov_vecs=sparseTestingData[1])
        else:
            train_cov_matrix=None
            test_cov_matrix=None
            
        phe_rs = PheRS()
        phe_rs.FitModel(sparseTrainingData[0],covariateMatrix=train_cov_matrix)
        trainScores = phe_rs.ComputeScores(sparseTrainingData[0],covariate_matrix=train_cov_matrix)
        testScores = phe_rs.ComputeScores(sparseTestingData[0],covariate_matrix=test_cov_matrix)
        
        if previousArrayType!='Sparse':
            datasetSampler.ChangeArrayType(previousArrayType)
        return trainScores,testScores
    
    def ComputeJaccardSimilarity(self,datasetSampler,index_vec):
        
        
        previousArrayType = datasetSampler.returnArrays
        if datasetSampler.returnArrays!='Sparse':
            datasetSampler.ChangeArrayType('Sparse')
        sparseTrainingData=datasetSampler.ReturnFullTrainingDataset(randomize=False)[0]
        index_vec_binary=np.zeros(sparseTrainingData.shape[1])
        index_vec_binary[index_vec]=1
        
        sparseTestingData=datasetSampler.ReturnFullTestingDataset(randomize=False)[0]
        jaccard = JaccardSimilarity(index_vec_binary)
        trainScores = jaccard.ComputeScores(sparseTrainingData)
        testScores = jaccard.ComputeScores(sparseTestingData)
        
        if previousArrayType!='Sparse':
            datasetSampler.ChangeArrayType(previousArrayType)
        return trainScores,testScores
    
    
    def ComputeEmbeddings(self,datasetSampler,num_components=1,embedding_type='NMF',includeCovariates=False):
        previousArrayType = datasetSampler.returnArrays
        if datasetSampler.returnArrays!='Sparse':
            datasetSampler.ChangeArrayType('Sparse')
        sparseTrainingData=datasetSampler.ReturnFullTrainingDataset(randomize=False)
        sparseTestingData=datasetSampler.ReturnFullTestingDataset(randomize=False)
        if includeCovariates:
            train_cov_matrix = datasetSampler.CollapseDataArrays(cov_vecs=sparseTrainingData[1])
            test_cov_matrix = datasetSampler.CollapseDataArrays(cov_vecs=sparseTestingData[1])
        else:
            train_cov_matrix=None
            test_cov_matrix=None
        
        embed=LinearEmbedding(num_components=num_components,embedding_model=embedding_type)
        
        embed.FitModel(sparseTrainingData[0],covariateMatrix=train_cov_matrix)
        trainEmbeddings = embed.ComputeEmbeddings(sparseTrainingData[0],train_cov_matrix)
        testEmbeddings = embed.ComputeEmbeddings(sparseTestingData[0],test_cov_matrix)
        if previousArrayType!='Sparse':
            datasetSampler.ChangeArrayType(previousArrayType)
        return trainEmbeddings,testEmbeddings
        
    def _readModelFromFile(self,fName):
        with open(fName,'rb') as f:
            model_dict = torch.load(f,map_location='cpu')
        return model_dict
    
    
    def LoadModel(self,stored_model):
        if not isinstance(stored_model,dict):
            assert isinstance(stored_model,str),"Expects file name if not provided with dictionary."
            stored_model = self._readModelFromFile(stored_model)
        assert set(stored_model.keys())==set(['model_state','posterior_params','prior_params','meta_data']),"Model dictionary must contain the following elements: 'model_state','posterior_params','prior_params','meta_data'"
        
        #first load meta data, make sure it aligns with structure of current vlpi instance
        stored_model_type = stored_model['meta_data']['modelType']
        assert stored_model_type==self.modelType,"Attempting to load disparate model types. Current model: {}, Stored model: {}".format(self.modelType,stored_model_type)
        if self.useAvailableAnchorDx is None:
            assert stored_model['meta_data']['useAvailableAnchorDx'] is None,"Current model is not expecting a anchor dx, while stored model was fit using this information."
        else:
            assert stored_model['meta_data']['useAvailableAnchorDx']==self.useAvailableAnchorDx,"Current and stored model have different anchor dx."

        
        if self.modelType=='Discrete':
            
            self.currentModel = DiscreteModel(stored_model['meta_data']['numObsTraits'], stored_model['meta_data']['numCatList'],self.useAvailableAnchorDx is not None,stored_model['meta_data']['mappingFunction'],**self.allmodel_kwargs)
            
    
                
        elif self.modelType=='Continuous':
            
            self.currentModel = ContinuousModel(stored_model['meta_data']['numObsTraits'], stored_model['meta_data']['numCatList'],self.useAvailableAnchorDx is not None,stored_model['meta_data']['nLatentDim'],stored_model['meta_data']['mappingFunction'],**self.allmodel_kwargs)
            
        else:
            self.currentModel = DiscriminativeModel(stored_model['meta_data']['numObsTraits'], stored_model['meta_data']['numCatList'],stored_model['meta_data']['mappingFunction'],linkFunction=stored_model['meta_data']['linkFunction'],coupleCovariates=stored_model['meta_data']['coupleCovariates'],**self.allmodel_kwargs)
            
        self.currentModel.LoadPriorState(stored_model)
            
    
            
    def PackageModel(self,fName=None):
        model_dict = self.currentModel.PackageCurrentState()
        model_dict['meta_data']={}
        model_dict['meta_data']['modelType']=self.modelType
        model_dict['meta_data']['useAvailableAnchorDx']=self.useAvailableAnchorDx
        model_dict['meta_data']['numObsTraits']=self.currentModel.numObsTraits
        model_dict['meta_data']['numCatList']=self.currentModel.numCatList
        model_dict['meta_data']['mappingFunction']=self.currentModel.mappingFunction
        if self.modelType=='Continuous':
            model_dict['meta_data']['nLatentDim']=self.currentModel.nLatentDim
        if self.modelType=='Discriminative':
            model_dict['meta_data']['linkFunction']=self.currentModel.linkFunctionType
            model_dict['meta_data']['coupleCovariates']=self.currentModel.coupleCovariates
            
        if fName is not None:
            with open(fName,'wb') as f:
                torch.save(model_dict,f)
        return model_dict

if __name__=='__main__':
    
    numSamples = 10000
    numAssociatedTraits=20
    numMaskedTraits=0
    numNoiseTraits=0
    useAnchorDx= False
    nLatentSimDim=2
    nLatentFitDim=2
    modelType = 'Continuous'
    simType = 'Continuous'
    mappingFunction='Linear_Monotonic'

#    numCovPerClass = [2,3,10] 
#    covNames = ['A','B','C']
    numCovPerClass = [] 
    covNames = []
    
    
    
    continuousModelParams={}
    continuousModelParams['nLatentDim']=nLatentSimDim
    continuousModelParams['anchorDxPriors']={'anchorDxNoise':[1.0,10.0],'latentDimToAnchorDxMap':0.1*nLatentSimDim,'prevalence':[0.00099,0.0011]}
    continuousModelParams['latentPhenotypePriors']={'element_wise_precision':[1,2.0]}
    continuousModelParams['covariatePriors']={'intercept':[-2,2.0],'cov_scale':0.025}
    
    discreteModelParams={}
    discreteModelParams['anchorDxPriors']={'anchorDxNoise':[1.0,10.0]}
    discreteModelParams['latentPhenotypePriors']={'element_wise_precision':[1,2.0],'prevalence':[1.0,10.0]}
    discreteModelParams['covariatePriors']={'intercept':[-2,2.0],'cov_scale':0.025}
    
    
    clinData = ClinicalDataset()
    
    disList = list(clinData.dxCodeToDataIndexMap.keys())[0:(numAssociatedTraits-numMaskedTraits)+numNoiseTraits+int(useAnchorDx)]
    clinData.IncludeOnly(disList)
    
    
    if useAnchorDx:
        useAvailableAnchorDx=disList[-1]
    else:
        useAvailableAnchorDx=None
        
    test_vlpi=vLPI(modelType,useAvailableAnchorDx=useAvailableAnchorDx,neuralNetworkHyperparameters={'n_layers' : 2, 'n_hidden' : 32, 'dropout_rate': 0.1, 'use_batch_norm':True})
    
    


    if simType=='Discrete':
        simData = test_vlpi.SimulateData(numSamples,numAssociatedTraits,numCategoricalCovariates=numCovPerClass,numMaskedTraits=numMaskedTraits,numNoiseTraits=numNoiseTraits,DiscreteModelParameters=discreteModelParams,simModelType='Discrete')
    else:
        simData = test_vlpi.SimulateData(numSamples,numAssociatedTraits,numCategoricalCovariates=numCovPerClass,numMaskedTraits=numMaskedTraits,numNoiseTraits=numNoiseTraits,ContinuousModelParameters=continuousModelParams,simModelType='Continuous')
    
    
    if useAnchorDx:
        clinData.LoadFromArrays(torch.cat((simData['incidence_data'],simData['anchor_dx_data']),dim=-1),simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')
        clinData.ConditionOnDx([useAvailableAnchorDx])
        sampler = ClinicalDatasetSampler(clinData,0.75,returnArrays='Torch',conditionSamplingOnDx=[useAvailableAnchorDx])
        
    else:
        clinData.LoadFromArrays(simData['incidence_data'],simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')
        sampler = ClinicalDatasetSampler(clinData,0.75,returnArrays='Torch')
        
    


    
    if modelType=='Continuous':
        continuousModelParams['nLatentDim']=nLatentFitDim
        output=test_vlpi.FitModel(sampler,batch_size=1000,mappingFunction='Linear_Monotonic',learningRate=0.05,withCosineAnnealing=True,initializationErrorTol=1e-2,errorTol=1e-4,maxEpochs=200,ContinuousModelParameters = continuousModelParams,numParticles=5,maxInitializationEpochs=50,alreadyInitialized=False)
    else:
        discreteModelParams['latentPhenotypePriors']={'element_wise_precision':[1,2.0],'prevalence':[1.0,1.0]}
        output=test_vlpi.FitModel(sampler,batch_size=1000,mappingFunction='Linear_Monotonic',learningRate=0.05,withCosineAnnealing=True,DiscreteModelParameters = discreteModelParams,initializationErrorTol=1e-3,errorTol=1e-4,maxEpochs=200,numParticles=10,maxInitializationEpochs=50,alreadyInitialized=False)
#        
    trainPheRSScores,testPheRSScores = test_vlpi.ComputePheRS(sampler,includeCovariates=False)
    trainEmbedScores,testEmbedScores =  test_vlpi.ComputeEmbeddings(sampler,num_components=1,includeCovariates=False)

    
    
    