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
from vlpi.UnsupervisedMethods import PhenotypeDecomposition
from vlpi._vlpiModel import _vlpiModel
from scipy.stats import gamma,norm

class vLPI:
        
    
    def _parseParamArgs(self,**kwargs):
        
        allKeywordArgs = list(kwargs.keys()) 
        
        if 'ModelParameters' not in allKeywordArgs:
            
            modelParameters={}
            modelParameters['latentPhenotypePriors']={'latentPhenotypeEffectsPrecision':[1.0,1.0]}
            modelParameters['fixedEffectPriors']={'intercepts':[0.0,5.0],'covariates_scale':3.0}
        else:
            assert isinstance(kwargs['ModelParameters'],dict),"Expect 'ModelParameters' to be a dictionary of paramters"
            assert set(kwargs['ModelParameters'].keys())==set(['latentPhenotypePriors','fixedEffectPriors']),"""ModelParameters must contain the following key-value pairs:
                KEY                      VALUE                
                latentPhenotypePriors    Dictionary with one entry: {'latentPhenotypeEffectsPrecision':[conc,rate]}. [conc,rate] represent gamma prior parameters specifying distribution over the precision of exponentially-distributed latent phenotype effects. 
                
                fixedEffectPriors:    Dictionary with two entries: {'intercepts':[mean,scale],'covariates_scale':scale}. Mean, scale represent mean and scale parameters of a gaussian prior over observed trait incidence. In cases in which only scale is provided, mean is assumed to be zero. Scale indicates scale for gaussian prior over covariate effects. Mean fixed at zero. 
                
                """
            modelParameters={}
            for key,value in kwargs['ModelParameters'].items():
                modelParameters[key]=value
        return modelParameters
    
 
    
    def __init__(self,datasetSampler,nLatentDim,**kwargs):
        """

        vLPI a statistical model that maps a latent phenotypic space to some vector of observed (binary) phenotypes through a function f(Z), where Z is the latent phenotypic space of interest. Z is assumed to be an multivariate gaussian prior with indentity covariance matrix (independent components). Inference of the model is conducted using a variational approximation of the model marginal model likelihood, which is estimated using gradient descent. To allow inference to scale to millions of patients and enable subsequent portability, latent phenotype inference is amoritized using a non-linear neural network.
    
        
        Formal mathematical definition of the model can be found in Blair et al ***.
        
        Arguments:
            datasetSampler: Instantiation of ClinicalDatasetSampler class for clinical dataset intended to model.
            
            nLatentDim: Number of latent dimensions to include in the model. Default is 1.
                        
            
        **kwargs: keyword arguments that can be passed to an arbitrary model. They consist of the following:
                        
            dropLinearCovariateColumn: boolean value indicated whether to drop one category from each covariate included into the model. Aids in identifiability. Defaults to True.
            
            neuralNetworkHyperparameters: speficies hyperparameters of the encoder/decoder neural networks. Default is a 2-layer MLP with 32 hidden nodes per layer. Dropout rate 0.1 with Batch-Normalization enabled. Larger networks are more difficult to train, while smaller networks are less expressive and may result in poorer approximations of the variational posterior.
            
            mappingFunction: Indicates function f(Z) that maps latent space to observed phenotypes. Currently, 'Linear' or 'Linear_Monotonic' supported.
            

        """
        self.sampler = datasetSampler
        if self.sampler.isConditioned:
            self.sampler.ConvertToUnconditional()
            print('Warning: vLPI passed a conditional ClinicalDatasetSampler. Converting to uncoditional sampler.')
 
        modelParameters=self._parseParamArgs(**kwargs)
            
        n_cat_list = [len(self.sampler.currentClinicalDataset.catCovConversionDicts[x]) for x in self.sampler.includedCovariates]
        
        
        self.model = _vlpiModel(self.sampler.currentClinicalDataset.numDxCodes, n_cat_list,nLatentDim,latentPhenotypePriors=modelParameters['latentPhenotypePriors'],fixedEffectPriors=modelParameters['fixedEffectPriors'],**kwargs)
        
        #(self,numObsTraits:int, numCatList:Iterable[int],nLatentDim:int,:str,latentPhenotypePriors={'latentPhenotypeEffectsPrecision':[1.0,1.0]},fixedEffectPriors={'intercepts':[0.0,5.0],'covariates_scale':3.0},**kwargs)
        
    
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


        if 'learningRate' in allKeywordArgs:
            learningRate=kwargs['learningRate']
        else:
            learningRate=0.04
            
        if 'errorTol' in allKeywordArgs:
            errorTol=kwargs['errorTol']
        else:
            errorTol = 1e-4
        
        if 'numParticles' in allKeywordArgs:
            numParticles=kwargs['numParticles']
        else:
            numParticles=10
            
            
        if 'maxEpochs' in allKeywordArgs:
            maxEpochs=kwargs['maxEpochs']
        else:
            maxEpochs=200
        
            
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
            withCosineAnealing=True
        
        cosineAnnealingDict = {'withCosineAnnealing':withCosineAnealing,'initialRestartInterval':10,'intervalMultiplier':2}


        optimizer=Optimizers(self.model,self.sampler,optimizationParameters={'initialLearningRate': learningRate,'maxEpochs': maxEpochs,'numParticles':numParticles},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},cosineAnnealing=cosineAnnealingDict)
        
        if alreadyInitialized==False:
            print("Initializing the encoder by training the against ICA-initialized NMF Model.")
            trainScores,testScores,components = self.ComputeNMF_Embeddings(num_components=self.model.nLatentDim)
                
            scoreVector = np.concatenate([trainScores,testScores])
            allSampleIndex = np.concatenate([self.sampler.trainingDataIndex,self.sampler.testDataIndex])
            self.sampler.AddScoresToDataset(scoreVector,allSampleIndex)
            
            

            if batch_size > 0:
                output=optimizer.BatchTrain(batch_size,errorTol=errorTol,optimizationStrategy='ScoreBased',verbose=verbose)
            else:
                output=optimizer.FullDatasetTrain(errorTol=errorTol,optimizationStrategy='ScoreBased',verbose=verbose)
            
            self.sampler.RemoveScoresFromDataset()
            print("Initializing the model by training against the NMF encoder.")
            
            
            if batch_size > 0:
                output=optimizer.BatchTrain(batch_size,errorTol=errorTol,optimizationStrategy='PosteriorOnly',verbose=verbose)
            else:
                output=optimizer.FullDatasetTrain(errorTol=errorTol,optimizationStrategy='PosteriorOnly',verbose=verbose)
            print("Re-initializing the encoder by training against the statistical model.")
                
            if batch_size > 0:
                output=optimizer.BatchTrain(batch_size,errorTol=errorTol,optimizationStrategy='EncoderOnly',verbose=verbose)
            else:
                output=optimizer.FullDatasetTrain(errorTol=errorTol,optimizationStrategy='EncoderOnly',verbose=verbose)
                
            print("Model successfully initialized. ELBO:{0:10.2f}".format(-1.0*output[0]))
        print("Optimizing full model.")
        if batch_size > 0:
            output=optimizer.BatchTrain(batch_size,errorTol=errorTol,optimizationStrategy='Full',verbose=verbose)
        else:
            output=optimizer.FullDatasetTrain(errorTol=errorTol,optimizationStrategy='Full',verbose=verbose)
        print('Inference complete. Final ELBO:{0:10.2f}'.format(-1.0*output[0]))
    
    def ComputeEmbeddings(self):
        previousArrayType = self.sampler.returnArrays
        if self.sampler.returnArrays!='Torch':
            self.sampler.ChangeArrayType('Torch')
        
        trainingData=self.sampler.ReturnFullTrainingDataset(randomize=False)
        testingData=self.sampler.ReturnFullTestingDataset(randomize=False)
        
        train_embed = self.model.PredictLatentPhenotypes(trainingData[0],trainingData[1])
        test_embed = self.model.PredictLatentPhenotypes(testingData[0],testingData[1])
        
        if previousArrayType!='Torch':
            self.sampler.ChangeArrayType(previousArrayType)
        return train_embed.detach().numpy(),test_embed.detach().numpy()
    
    def ComputeEuclidianDistanceScore(self):
        r
    
    def ReturnComponents(self,includeCI=False):
        
        if self.model.mappingFunction=='Linear_Monotonic':
            
            mean = (self.model.posteriorParamDict['latentPhenotypeEffects']['conc']/test_vlpi.model.posteriorParamDict['latentPhenotypeEffects']['rate']).detach().numpy()
            
            if includeCI:
                 
                dist = gamma(self.model.posteriorParamDict['latentPhenotypeEffects']['conc'].detach().numpy(),scale = 1.0/self.model.posteriorParamDict['latentPhenotypeEffects']['rate'].detach().numpy())
                
        else:
            mean = self.model.posteriorParamDict['latentPhenotypeEffects']['mean'].detach().numpy()
            
            if includeCI:
                dist = norm(loc=self.model.posteriorParamDict['latentPhenotypeEffects']['mean'].detach().numpy(),scale = self.model.posteriorParamDict['latentPhenotypeEffects']['scale'].detach().numpy())
                
        if includeCI:
            low = dist.ppf(0.025)
            high = dist.ppf(0.975)
            ci = np.concatenate([low[:,:,np.newaxis],high[:,:,np.newaxis]],axis=2)
            return mean,ci
        else:
            return mean
            
        
        
        
    
    def ComputeNMF_Embeddings(self,num_components=1,includeCovariates=False):
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
        
        embed=PhenotypeDecomposition(num_components=num_components)
        
        trainEmbeddings,centroid=embed.FitModel(sparseTrainingData[0],covariateMatrix=train_cov_matrix)
        testEmbeddings,dist = embed.ComputeScores(sparseTestingData[0],test_cov_matrix)
        if previousArrayType!='Sparse':
            self.sampler.ChangeArrayType(previousArrayType)
        return trainEmbeddings,testEmbeddings,embed.ReturnComponents()
        
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

        self.currentModel = _vlpiModel(stored_model['meta_data']['numObsTraits'], stored_model['meta_data']['numCatList'],stored_model['meta_data']['nLatentDim'],stored_model['meta_data']['mappingFunction'],**self.allmodel_kwargs)
                        
        self.currentModel.LoadPriorState(stored_model)
    

            
    def PackageModel(self,fName=None):
        model_dict = self.currentModel.PackageCurrentState()
        model_dict['meta_data']={}
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
    
    numSamples = 100000
    numAssociatedTraits=20
    nLatentSimDim=4
    nLatentFitDim=4
    simulateLabels=False
    mappingFunction='Linear_Monotonic'

#    numCovPerClass = [2,3,10] 
#    covNames = ['A','B','C']
    numCovPerClass = [] 
    covNames = []
    
    
    
    modelParams={}
    modelParams['latentPhenotypePriors']={'latentPhenotypeEffectsPrecision':[1,10.0]}
    modelParams['fixedEffectPriors']={'intercepts':[-2,2.0],'covariates_scale':0.025}
    
    simulator = ClinicalDataSimulator(numAssociatedTraits,nLatentSimDim)
    simData=simulator.GenerateClinicalData(numSamples,0.0)
    
    clinData = ClinicalDataset()
    
    disList = list(clinData.dxCodeToDataIndexMap.keys())[0:numAssociatedTraits+int(simulateLabels)]
    clinData.IncludeOnly(disList)
    
    
    if simulateLabels:
        labelDx=disList[-1]
    else:
        labelDx=None
        
        
    if simulateLabels:
        clinData.LoadFromArrays(torch.cat((simData['incidence_data'],simData['label_dx_data']),dim=-1),simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')
        clinData.ConditionOnDx([labelDx])
        sampler = ClinicalDatasetSampler(clinData,0.75,returnArrays='Torch',conditionSamplingOnDx=[labelDx])
        
    else:
        clinData.LoadFromArrays(simData['incidence_data'],simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')
        sampler = ClinicalDatasetSampler(clinData,0.75,returnArrays='Torch')
        
    test_vlpi=vLPI(sampler,nLatentFitDim,ModelParameters=modelParams,mappingFunction=mappingFunction,neuralNetworkHyperparameters={'n_layers' : 2, 'n_hidden' : 128, 'dropout_rate': 0.2, 'use_batch_norm':True})
    output=test_vlpi.FitModel(batch_size=1000)
    nmf_train_scores, nmf_test_scores, nmf_components=test_vlpi.ComputeNMF_Embeddings(nLatentFitDim)
    
    model_train_scores,model_test_scores = test_vlpi.ComputeEmbeddings()
    model_components = test_vlpi.ReturnComponents()
    
    
#    test_vlpi_4=vLPI(sampler,4,ModelParameters=modelParams,neuralNetworkHyperparameters={'n_layers' : 2, 'n_hidden' : 32, 'dropout_rate': 0.1, 'use_batch_norm':True})
#    output_4=test_vlpi_4.FitModel(batch_size=1000,learningRate=0.05,initializationErrorTol=1e-2,withCosineAnnealing=False)
#
#    embed_4=test_vlpi_4.ComputeEmbeddings(nLatentFitDim)
#    
    


    
    

        
    
    
    
    