#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:18:43 2019

@author: davidblair
"""

import torch
import pyro
import numpy as np
from vlpi.optim.Optimizer import Optimizer
from vlpi.data.ClinicalDataset import ClinicalDatasetSampler,ClinicalDataset
from vlpi.utils.UnsupervisedMethods import PhenotypeDecomposition
from vlpi.model.VAE import VAE

class vLPI:


    def __init__(self,datasetSampler,nLatentDim,latentPhenotypeMap='Linear_Monotonic',**kwargs):
        """

        vLPI a statistical model that maps a latent phenotypic space to some vector of observed (binary) phenotypes through a function f(Z), where Z is the latent phenotypic space of interest. Z is assumed to be an multivariate gaussian prior with indentity covariance matrix (independent components). Inference of the model is conducted using a variational approximation of the model marginal model likelihood, which is estimated using gradient descent. To allow inference to scale to millions of patients and enable subsequent portability, latent phenotype inference is amoritized using a non-linear neural network.


        Formal definition of the model can be found in Blair et al ***.

        Arguments:
            datasetSampler: Instantiation of ClinicalDatasetSampler class for clinical dataset intended to model.

            nLatentDim: Number of latent dimensions to include in the model. Default is 1.
            latentPhenotypeMap: Function f(Z) mapping latent to observed phenotypes. Must be one of the following ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic']. Defaults to 'Linear_Monotonic'


        **kwargs: keyword arguments that can be passed to an arbitrary model. They consist of the following:

            dropLinearCovariateColumn: boolean value indicated whether to drop one category from each covariate included into the model. Aids in identifiability. Defaults to True.

            encoderNetworkHyperparameters: speficies hyperparameters of the encoder neural networks. Default is a 2-layer MLP with 128 hidden nodes per layer. Dropout rate 0.2 with Batch-Normalization enabled. Larger networks are more difficult to train, while smaller networks are less expressive and may result in poorer approximations of the variational posterior.


        """
        self.sampler = datasetSampler
        if self.sampler.isConditioned:
            self.sampler.ConvertToUnconditional()
            print('Warning: vLPI passed a conditional ClinicalDatasetSampler. Converting to unconditioned sampler.')

        self.all_model_kwargs = kwargs

        n_cat_list = [len(self.sampler.currentClinicalDataset.catCovConversionDicts[x]) for x in self.sampler.includedCovariates]

        assert latentPhenotypeMap in ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'], "Currently supported latent-to-observed phenotype maps include: 'Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'"
        
        if latentPhenotypeMap!='Linear_Monotonic':
            print('WARNING: Software is currently optimized for the Linear_Monotonic mapping function. Other mapping functions are experimental, as model initialization and optimization parameters were specifically chosen with this mapping function in mind. Properly optimizing these other functions may require modification to initialization routine AND/OR experimenting with optimization hyperparameters.')
        
        self.model = VAE(self.sampler.currentClinicalDataset.numDxCodes, n_cat_list,nLatentDim,latentPhenotypeMap,**kwargs)




    def FitModel(self,batch_size=0,verbose=True,**kwargs):
        """

        batch_size: size of mini-batches for stochastic inference. Default is 0, which indicates that the full dataset will be used.
        verbose: indicates whether or not to print out inference progress in real time

        Keyword Arguments:
            alreadyInitialized: boolean, indicates whether the model has been pre-initialized (by copying from previous state). This way initialization procedure can be ignored. Can also use to skip initialization procedure on un-initialized model, at which point it is initialize witht the prior.


            learningRate: initial learning rate for the stochastic gradient descent. Default is 0.005.

            withCosineAnealing: boolean, indicates whether to use cosine annealing with warm restarts to adjust learning rate during inference. Default is False.

            numParticles: number of particles used to approximate gradient during inference. Default is 10.

            errorTol: errorTolerance for general model fitting. Default is 1e-4

            maxEpochs: Maximum number of epochs for model fitting. Default is 1000

            computeDevice: Device to use for model fitting. Default is None, which specifies cpu. Use integer (corresponding to device number) to specify gpu.

            numDataLoaders: number of dataloaders used for gpu computing, as model fitting becomes I/O bound if using gpu. If not gpu enabled, must be 0.
        """

        ######### Parse Keyword Arguments #########
        allKeywordArgs = list(kwargs.keys())

        #allows user to keep the param store the same if re-starting optimization, for example, after loading from disk
        if 'alreadyInitialized' in allKeywordArgs:
            alreadyInitialized=kwargs['alreadyInitialized']
        else:
            alreadyInitialized=False


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


        optimizer=Optimizer(self.model,self.sampler,optimizationParameters={'initialLearningRate': learningRate,'maxEpochs': maxEpochs,'numParticles':numParticles},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},cosineAnnealing=cosineAnnealingDict)
        pyro.clear_param_store()


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
            print("Initializing the decoder by training against the NMF encoder.")


            if batch_size > 0:
                output=optimizer.BatchTrain(batch_size,errorTol=errorTol,optimizationStrategy='DecoderOnly',verbose=verbose)
            else:
                output=optimizer.FullDatasetTrain(errorTol=errorTol,optimizationStrategy='DecoderOnly',verbose=verbose)
            print("Re-initializing the encoder by training against the decoder.")

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
        return -1.0*output[0]

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

    def ComputeOutlierScores(self):
        trainEmbeddings,testEmbeddings = self.ComputeEmbeddings()
        centroid = np.mean(trainEmbeddings,axis=0)
        return np.sqrt(np.sum((testEmbeddings-centroid)**2.0,axis=1))


    def ReturnComponents(self,includeCI=False):

        assert self.model.decoderType in ['Linear_Monotonic','Linear'],"Components only availabe for Linear models. vLPI fit using a non-linear mapping function."
        mapping_func_state_dict = self.model.decoder.state_dict()

        if self.model.decoderType == 'Linear_Monotonic':
            return np.exp(mapping_func_state_dict['linear_latent.log_scale_weight'].detach().numpy()).T
        else:
            return mapping_func_state_dict['linear_latent.weight'].detach().numpy().T




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
    
    def ComputePerplexity(self):
        previousArrayType = self.sampler.returnArrays
        if self.sampler.returnArrays!='Torch':
            self.sampler.ChangeArrayType('Torch')

        trainingData=self.sampler.ReturnFullTrainingDataset(randomize=False)
        testingData=self.sampler.ReturnFullTestingDataset(randomize=False)

        train_elbo = self.model.ComputeELBOPerDatum(trainingData[0],trainingData[1])
        test_elbo = self.model.ComputeELBOPerDatum(testingData[0],testingData[1])

        if previousArrayType!='Torch':
            self.sampler.ChangeArrayType(previousArrayType)
        return -1.0*train_elbo.detach().numpy(),-1.0*test_elbo.detach().numpy()
        
        

    def _readModelFromFile(self,fName):
        with open(fName,'rb') as f:
            model_dict = torch.load(f,map_location='cpu')
        return model_dict


    def LoadModel(self,stored_model):
        if not isinstance(stored_model,dict):
            assert isinstance(stored_model,str),"Expects file name if not provided with dictionary."
            stored_model = self._readModelFromFile(stored_model)
        assert set(stored_model.keys())==set(['model_state','meta_data']),"Model dictionary must contain the following elements: 'model_state','meta_data'"

        #first load meta data, make sure it aligns with structure of current vlpi instance

        self.model = VAE(stored_model['meta_data']['numObsTraits'], stored_model['meta_data']['numCatList'],stored_model['meta_data']['nLatentDim'],stored_model['meta_data']['latentPhenotypeMap'],**self.all_model_kwargs)
        self.model.LoadPriorState(stored_model)



    def PackageModel(self,fName=None):
        model_dict = self.model.PackageCurrentState()
        model_dict['meta_data']={}
        model_dict['meta_data']['numObsTraits']=self.model.numObsTraits
        model_dict['meta_data']['numCatList']=self.model.numCatList
        model_dict['meta_data']['latentPhenotypeMap']=self.model.decoderType
        model_dict['meta_data']['nLatentDim']=self.model.nLatentDim

        if fName is not None:
            with open(fName,'wb') as f:
                torch.save(model_dict,f)
        return model_dict

if __name__=='__main__':

    from vlpi.data.ClinicalDataSimulator import ClinicalDataSimulator
    
    pyro.clear_param_store()

    numSamples = 1000
    numAssociatedTraits=20
    nLatentSimDim=4
    nLatentFitDim=4
    simulateLabels=False
    mappingFunction='Linear_Monotonic'

    numCovPerClass = [2,3,10]
    covNames = ['A','B','C']
#    numCovPerClass = []
#    covNames = []



    modelParams={}
    modelParams['latentPhenotypePriors']={'latentPhenotypeEffectsPrecision':[1,10.0]}
    modelParams['fixedEffectPriors']={'intercepts':[-2,2.0],'covariates_scale':0.025}

    simulator = ClinicalDataSimulator(numAssociatedTraits,nLatentSimDim,numCovPerClass)
    simData=simulator.GenerateClinicalData(numSamples,0.01)

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

    test_vlpi=vLPI(sampler,nLatentFitDim)
    output=test_vlpi.FitModel(batch_size=100)
    nmf_train_scores, nmf_test_scores, nmf_components=test_vlpi.ComputeNMF_Embeddings(nLatentFitDim)

    model_train_scores,model_test_scores = test_vlpi.ComputeEmbeddings()
    model_components = test_vlpi.ReturnComponents()

    euclid_score = test_vlpi.ComputeOutlierScores()
    perplex_train,perplex_test = test_vlpi.ComputePerplexity()
    
#    test_vlpi.PackageModel('/Users/davidblair/Desktop/tmp.pth')
    
#    pyro.clear_param_store()
#    new_vlpi = vLPI(sampler,nLatentFitDim)
#    new_vlpi.LoadModel('/Users/davidblair/Desktop/tmp.pth')
##    
#    output=new_vlpi.FitModel(batch_size=100,alreadyInitialized = True)
    


#    test_vlpi_4=vLPI(sampler,4,ModelParameters=modelParams,neuralNetworkHyperparameters={'n_layers' : 2, 'n_hidden' : 32, 'dropout_rate': 0.1, 'use_batch_norm':True})
#    output_4=test_vlpi_4.FitModel(batch_size=1000,learningRate=0.05,initializationErrorTol=1e-2,withCosineAnnealing=False)
#
#    embed_4=test_vlpi_4.ComputeEmbeddings(nLatentFitDim)
#
