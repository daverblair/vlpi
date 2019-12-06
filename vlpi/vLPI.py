#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:18:43 2019

@author: davidblair
"""

import torch
import pyro
import numpy as np
from vlpi.Optimizers import Optimizers
from vlpi.ClinicalDataset import ClinicalDatasetSampler,ClinicalDataset
from vlpi.UnsupervisedMethods import LinearEmbedding
from vlpi._vlpiModel import _vlpiModel
from vlpi.utils import rel_diff


class vLPI:
        
    
    def _parseParamArgs(self,**kwargs):
        
        allKeywordArgs = list(kwargs.keys()) 
        
        if 'ModelParameters' not in allKeywordArgs:
            
            modelParameters={}
            modelParameters['anchorDxPriors']={'anchorDxNoise':[1.0,1.0],'latentDimToAnchorDxMap':1.0,'anchorDxPrevalence':[1e-6,0.001]}
            modelParameters['latentPhenotypePriors']={'latentPhenotypeEffectsPrecision':[1.0,1.0]}
            modelParameters['fixedEffectPriors']={'intercepts':[0.0,5.0],'covariates_scale':3.0}
        else:
            assert isinstance(kwargs['ModelParameters'],dict),"Expect 'ModelParameters' to be a dictionary of paramters"
            assert set(kwargs['ModelParameters'].keys())==set(['anchorDxPriors','latentPhenotypePriors','fixedEffectPriors']),"""ModelParameters must contain the following key-value pairs:
                KEY                      VALUE
                anchorDxPriors           Dictionary with three entries: {'anchorDxNoise':[conc,rate],'anchorDxPrevalence':[low,high],'latentDimToAnchorDxMap':alpha}, where [conc,rate] denote gamma prior parameters over gaussian distributed noise, [low,high] represent low and high ends of a 95% CI that represents the prior over anchor phenotype prevalence (each value must be between 0.0 and 1.0; can be VERY broad), and alpha denotes the concentration prior for a symmetric dirichlet.
                
                latentPhenotypePriors    Dictionary with one entry: {'latentPhenotypeEffectsPrecision':[conc,rate]}. [conc,rate] represent gamma prior parameters specifying distribution over the precision of exponentially-distributed latent phenotype effects. 
                
                fixedEffectPriors:    Dictionary with two entries: {'intercepts':[mean,scale],'covariates_scale':scale}. Mean, scale represent mean and scale parameters of a gaussian prior over observed trait incidence. In cases in which only scale is provided, mean is assumed to be zero. Scale indicates scale for gaussian prior over covariate effects. Mean fixed at zero. 
                
                """
            modelParameters={}
            for key,value in kwargs['ModelParameters'].items():
                modelParameters[key]=value
        return modelParameters
    
 
    
    def __init__(self,datasetSampler,nLatentDim=1,**kwargs):
        """

        vLPI a statistical model that maps a latent phenotypic space to some vector of observed (binary) phenotypes through a function f(Z), where Z is the latent phenotypic space of interest. Z is assumed to be an multivariate gaussian prior with indentity covariance matrix (independent components). Inference of the model is conducted using a variational approximation of the model marginal model likelihood, which is estimated using gradient descent. To allow inference to scale to millions of patients and enable subsequent portability, latent phenotype inference is amoritized using a non-linear neural network.
        
        
        To improve inferential accuracy,the model can also use information provided in the form of labels marking the extreme end of the latent phenotypic spectrum, which we call anchor diagnoses. This can be thought of as the noisy output of some measurement of the latent phenotypic state. In such settings, the model corresponds to a form of "noisy" supervised inference.
        
        Formal mathematical definition of the model can be found in Blair et al ***.
        
        Arguments:
            datasetSampler: Instantiation of ClinicalDatasetSampler class for clinical dataset intended to model.
            
            nLatentDim: Number of latent dimensions to include in the model. Default is 1.
                        
            
        **kwargs: keyword arguments that can be passed to an arbitrary model. They consist of the following:
                        
            dropLinearCovariateColumn: boolean value indicated whether to drop one category from each covariate included into the model. Aids in identifiability. Defaults to True.
            
            neuralNetworkHyperparameters: speficies hyperparameters of the encoder/decoder neural networks. Default is a 2-layer MLP with 32 hidden nodes per layer. Dropout rate 0.1 with Batch-Normalization enabled. Larger networks are more difficult to train, while smaller networks are less expressive and may result in poorer approximations of the variational posterior.
            
            mappingFunction: Indicates function f(Z) that maps latent space to observed phenotypes. Currently, 'Linear' or 'Linear_Monotonic' supported.
            

        """
        allmodel_kwargs = list(kwargs.keys()) 
        self.sampler = datasetSampler
        
        if self.sampler.isConditioned:
            assert len(self.sampler.conditionSamplingOnDx)==1,"Model currently supports only a single anchoring diagnosis."
            self.anchorDx = self.sampler.conditionSamplingOnDx[0]
        else:
            self.anchorDx=None
            
        if 'mappingFunction' in allmodel_kwargs:
                mappingFunction=kwargs['mappingFunction']
        else:
            mappingFunction='Linear_Monotonic'
            
        modelParameters=self._parseParamArgs(**kwargs)
            
        n_cat_list = [len(self.sampler.currentClinicalDataset.catCovConversionDicts[x]) for x in self.sampler.includedCovariates]
        
        self.model = _vlpiModel(self.sampler.currentClinicalDataset.numDxCodes, n_cat_list,self.anchorDx is not None,nLatentDim,mappingFunction,anchorDxPriors=modelParameters['anchorDxPriors'],latentPhenotypePriors=modelParameters['latentPhenotypePriors'],covariatePriors=modelParameters['fixedEffectPriors'],**kwargs)
        
        
 
        
    
    def FitModel(self,batch_size=0,verbose=True,**kwargs):
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


        if 'statisticalModelLearningRate' in allKeywordArgs:
            statModelLR=kwargs['statisticalModelLearningRate']
        else:
            statModelLR=0.05
            
        if 'neuralNetLearningRate' in allKeywordArgs:
            neuralNetLR=kwargs['neuralNetLearningRate']
        else:
            neuralNetLR=0.05
        
        if 'numParticles' in allKeywordArgs:
            numParticles=kwargs['numParticles']
        else:
            numParticles=10
            

        if 'initializationErrorTol' in allKeywordArgs:
            initializationErrorTol=kwargs['initializationErrorTol']
        else:
            initializationErrorTol = 1e-2
            
        if 'maxInitializationEpochs' in allKeywordArgs:
            maxInitializationEpochs=kwargs['maxInitializationEpochs']
        else:
            maxInitializationEpochs=20
        
        if 'refinementErrorTol' in allKeywordArgs:
            refinementErrorTol=kwargs['refinementErrorTol']
        else:
            refinementErrorTol=1e-4

        if 'maxRefinementEpochs' in allKeywordArgs:
            maxRefinementEpochs = kwargs['maxRefinementEpochs']
        else:
            maxRefinementEpochs = 100
            
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
            
        if 'withCosineAnnealing' in allKeywordArgs:
            withCosineAnealing=kwargs['withCosineAnnealing']
        else:
            withCosineAnealing=False
        
        cosineAnnealingDict = {'withCosineAnnealing':withCosineAnealing,'initialRestartInterval':10,'intervalMultiplier':2}


        if alreadyInitialized==False:
            print("Initializing the model by training the encoder against ICA-initialized NMF Model")
            trainScores,testScores = self.ComputeEmbeddings(num_components=self.model.nLatentDim)
                
            scoreVector = np.concatenate([trainScores,testScores])
                            
            if self.sampler.isConditioned:
                allSampleIndex = np.concatenate([np.concatenate(self.sampler.trainingDataIndex),np.concatenate(self.sampler.testDataIndex)])
            else:
                allSampleIndex = np.concatenate([self.sampler.trainingDataIndex,self.sampler.testDataIndex])
            
            self.sampler.AddScoresToDataset(scoreVector,allSampleIndex)
#            
##            
##            #initialize encoder bu matching to a rank based encoder
            optimizer_encoder=Optimizers(self.model,self.sampler,optimizationParameters={'initialLearningRate': neuralNetLR,'maxEpochs': maxInitializationEpochs,'numParticles':numParticles},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},cosineAnnealing=cosineAnnealingDict)
            
            
            optimizer_posterior=Optimizers(self.model,self.sampler,optimizationParameters={'initialLearningRate': statModelLR,'maxEpochs': maxInitializationEpochs,'numParticles':numParticles},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},cosineAnnealing=cosineAnnealingDict)
            
            
            if batch_size > 0:
                output=optimizer_encoder.BatchTrain(batch_size,errorTol=initializationErrorTol,optimizationStrategy='ScoreBased',verbose=False)
            else:
                output=optimizer_encoder.FullDatasetTrain(errorTol=initializationErrorTol,optimizationStrategy='ScoreBased',verbose=False)
            self.sampler.RemoveScoresFromDataset()
            
            if batch_size > 0:
                output=optimizer_posterior.BatchTrain(batch_size,errorTol=initializationErrorTol,optimizationStrategy='PosteriorOnly',verbose=False)
            else:
                output=optimizer_posterior.FullDatasetTrain(errorTol=initializationErrorTol,optimizationStrategy='PosteriorOnly',verbose=False)
                
            
            
            print("Now performing alternating optimization of encoder and variational posterior")
            
            currentELBO = self.model.ComputeELBO()
            
            
            for i in range(maxInitializationEpochs):
                if batch_size > 0:
                    output=optimizer_posterior.BatchTrain(batch_size,errorTol=initializationErrorTol,optimizationStrategy='PosteriorOnly',verbose=verbose)
                else:
                    output=optimizer_posterior.FullDatasetTrain(errorTol=initializationErrorTol,optimizationStrategy='PosteriorOnly',verbose=verbose)
                    
                if batch_size > 0:
                    output=optimizer_encoder.BatchTrain(batch_size,errorTol=initializationErrorTol,optimizationStrategy='EncoderOnly',verbose=verbose)
                else:
                    output=optimizer_encoder.FullDatasetTrain(errorTol=initializationErrorTol,optimizationStrategy='EncoderOnly',verbose=verbose)
#
                newELBO = output[0]
                
                
            optimizer=Optimizers(self.model,self.sampler,optimizationParameters={'initialLearningRate': neuralNetLR,'maxEpochs': maxRefinementEpochs,'numParticles':numParticles},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},cosineAnnealing=cosineAnnealingDict)

            if batch_size > 0:
                output=optimizer.BatchTrain(batch_size,errorTol=refinementErrorTol,optimizationStrategy='Full',verbose=verbose)
            else:
                output=optimizer.FullDatasetTrain(errorTol=refinementErrorTol,optimizationStrategy='Full',verbose=verbose)
            
        return output
    
    
    
    def ComputeEmbeddings(self,num_components=1,embedding_type='NMF',includeCovariates=False):
        previousArrayType = self.sampler.returnArrays
        if self.sampler.returnArrays!='Sparse':
            self.sampler.ChangeArrayType('Sparse')
        sparseTrainingData=self.sampler.ReturnFullTrainingDataset(randomize=False)
        sparseTestingData=self.sampler.ReturnFullTestingDataset(randomize=False)
        if includeCovariates:
            train_cov_matrix = self.sampler.CollapseDataArrays(cov_vecs=sparseTrainingData[1])
            test_cov_matrix = self.sampler.CollapseDataArrays(cov_vecs=sparseTestingData[1])
        else:
            train_cov_matrix=None
            test_cov_matrix=None
        
        embed=LinearEmbedding(num_components=num_components,embedding_model=embedding_type)
        
        embed.FitModel(sparseTrainingData[0],covariateMatrix=train_cov_matrix)
        trainEmbeddings = embed.ComputeEmbeddings(sparseTrainingData[0],train_cov_matrix)
        testEmbeddings = embed.ComputeEmbeddings(sparseTestingData[0],test_cov_matrix)
        if previousArrayType!='Sparse':
            self.sampler.ChangeArrayType(previousArrayType)
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
        if self.anchorDx is None:
            assert stored_model['meta_data']['anchorDx'] is None,"Current model is not expecting a anchor dx, while stored model was fit using this information."
        else:
            assert stored_model['meta_data']['anchorDx']==self.anchorDx,"Current and stored model have different anchor dx."
            

        self.currentModel = _vlpiModel(stored_model['meta_data']['numObsTraits'], stored_model['meta_data']['numCatList'],self.anchorDx is not None,stored_model['meta_data']['nLatentDim'],stored_model['meta_data']['mappingFunction'],**self.allmodel_kwargs)
                        
        self.currentModel.LoadPriorState(stored_model)
    
            
    def PackageModel(self,fName=None):
        model_dict = self.currentModel.PackageCurrentState()
        model_dict['meta_data']={}
        model_dict['meta_data']['anchorDx']=self.anchorDx
        model_dict['meta_data']['numObsTraits']=self.currentModel.numObsTraits
        model_dict['meta_data']['numCatList']=self.currentModel.numCatList
        model_dict['meta_data']['mappingFunction']=self.currentModel.mappingFunction
        model_dict['meta_data']['nLatentDim']=self.currentModel.nLatentDim
            
        if fName is not None:
            with open(fName,'wb') as f:
                torch.save(model_dict,f)
        return model_dict

if __name__=='__main__':
    
    from vlpi.ClinicalDataSimulator import ClinicalDataSimulator
    
    numSamples = 10000
    numAssociatedTraits=10
    useAnchorDx= False
    nLatentSimDim=2
    nLatentFitDim=4
    mappingFunction='Linear_Monotonic'

#    numCovPerClass = [2,3,10] 
#    covNames = ['A','B','C']
    numCovPerClass = [] 
    covNames = []
    
    
    
    modelParams={}
    modelParams['anchorDxPriors']={'anchorDxNoise':[1.0,10.0],'latentDimToAnchorDxMap':1.0,'anchorDxPrevalence':[0.001,0.01]}
    modelParams['latentPhenotypePriors']={'latentPhenotypeEffectsPrecision':[1,2.0]}
    modelParams['fixedEffectPriors']={'intercepts':[-2,2.0],'covariates_scale':0.025}
    
    simulator = ClinicalDataSimulator(numAssociatedTraits,nLatentSimDim,sparsityRate = 0.5)
    simData=simulator.GenerateClinicalData(numSamples)
    
    clinData = ClinicalDataset()
    
    disList = list(clinData.dxCodeToDataIndexMap.keys())[0:numAssociatedTraits+int(useAnchorDx)]
    clinData.IncludeOnly(disList)
    
    
    if useAnchorDx:
        anchorDx=disList[-1]
    else:
        anchorDx=None
        
        
    if useAnchorDx:
        anchorDx=disList[-1]
        clinData.LoadFromArrays(torch.cat((simData['incidence_data'],simData['anchor_dx_data']),dim=-1),simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')
        clinData.ConditionOnDx([anchorDx])
        sampler = ClinicalDatasetSampler(clinData,0.75,returnArrays='Torch',conditionSamplingOnDx=[anchorDx])
        
    else:
        
        clinData.LoadFromArrays(simData['incidence_data'],simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')
        sampler = ClinicalDatasetSampler(clinData,0.75,returnArrays='Torch')
        anchorDx=None
        
    test_vlpi_2=vLPI(sampler,2,ModelParameters=modelParams,neuralNetworkHyperparameters={'n_layers' : 2, 'n_hidden' : 32, 'dropout_rate': 0.1, 'use_batch_norm':True})
    output_2=test_vlpi_2.FitModel(batch_size=1000,learningRate=0.05,initializationErrorTol=1e-2,withCosineAnnealing=False)
    embed_2=test_vlpi_2.ComputeEmbeddings(nLatentFitDim)
    
#    test_vlpi_4=vLPI(sampler,4,ModelParameters=modelParams,neuralNetworkHyperparameters={'n_layers' : 2, 'n_hidden' : 32, 'dropout_rate': 0.1, 'use_batch_norm':True})
#    output_4=test_vlpi_4.FitModel(batch_size=1000,learningRate=0.05,initializationErrorTol=1e-2,withCosineAnnealing=False)
#
#    embed_4=test_vlpi_4.ComputeEmbeddings(nLatentFitDim)
#    
    


    
    

        
    
    
    
    