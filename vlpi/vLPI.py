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
from vlpi.model.VAE import VAE

__version__ = "0.1.7"

class vLPI:



    def __init__(self,datasetSampler,nLatentDim,latentPhenotypeMap='Linear_Monotonic',**kwargs):
        """
        vLPI (variational-Latent Phenotype Inference) is a statistical model that maps a latent phenotypic space to some vector of observed (binary) traits through a function f(Z), where Z is the latent phenotypic space of interest. Z is assumed to be an isotropic multivariate gaussian distribution, which is equivalent to assuming independence among the latent components. Inference of the model is conducted using a variational approximation of the model marginal model likelihood, which is optimized through stochastic gradient descent. To allow inference to scale to millions of patients and enable subsequent portability, latent phenotype inference is amortized using a non-linear neural network.

        Parameters
        ----------
        datasetSampler : ClinicalDatasetSampler
            Class that contains the ClinicalData and a mechanism for (efficiently) generating random samples.
        nLatentDim : int
            Number of latent phenotypes to include into the model.
        latentPhenotypeMap : str, optional
            Function f(Z) mapping latent to observed phenotypes. Must be one of the following ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic']. The default is 'Linear_Monotonic'. Initial testing has only been performed using this function.
        **kwargs: These are arguments passed to the class VAE, which is the modeling class (writtent in PyTorch and pyro) used for inference.

            linkFunction: Function used to map risk-scores produced by f(Z) to the probability scale [0.0,1.0]. Must be in ['Logit','Probit']. Note, 'Logit' is considerably faster.

            dropLinearCovariateColumn: boolean value indicated whether to drop one category from each covariate included into the model. Aids in identifiability. Defaults to True.

            encoderNetworkHyperparameters: dictionary that specifies hyperparameters of the encoder neural networks. Default is a 2-layer MLP with 64 hidden nodes per layer. Dropout rate 0.0 with Batch-Normalization enabled. Larger networks are more difficult to train, while smaller networks are less expressive and may result in poorer approximations of the variational posterior. Based on testing, allowing for dropout appears to limit the models ability to detect low prevalence components. Default: {'n_layers' : 2, 'n_hidden' : 64, 'dropout_rate': 0.0, 'use_batch_norm':True}

            coupleCovariates: Specifies whether to couple covariates to non-linear MLP network (True), or to model them  using an independent linear network (False). Defaults to True. Only relevant if using a Non-linear decoder ('Nonlinear' or 'Nonlinear_Monotonic')

            decoderNetworkHyperparameters: If using a non-linear symptom risk function (f(Z)), this dictionary specifies hyperparameters. Default matches the encoder hyperparameters: {'n_layers' : 2, 'n_hidden' : 64, 'dropout_rate': 0.0, 'use_batch_norm':True}



        Returns
        -------
        None.

        """

        self.sampler = datasetSampler
        if self.sampler.isConditioned:
            self.sampler.ConvertToUnconditional()
            print('Warning: vLPI passed a conditional ClinicalDatasetSampler. Converting to unconditioned sampler.')

        self.all_model_kwargs = kwargs

        n_cat_list = [len(self.sampler.currentClinicalDataset.catCovConversionDicts[x]) for x in self.sampler.includedCovariates]

        assert latentPhenotypeMap in ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'], "Currently supported latent-to-observed phenotype maps include: 'Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'"

        if latentPhenotypeMap!='Linear_Monotonic':
            print('WARNING: Software is currently optimized for the Linear_Monotonic mapping function. Other mapping functions are experimental, as model optimization parameters were specifically chosen with this mapping function in mind. Properly optimizing these other functions may require experimenting with optimization hyperparameters.')

        self.model = VAE(self.sampler.currentClinicalDataset.numDxCodes, n_cat_list,nLatentDim,latentPhenotypeMap,**kwargs)




    def FitModel(self,batch_size=1000,verbose=True,**kwargs):
        """


        Parameters
        ----------
        batch_size : int, optional
            Size of dataset batches for inference. The default is 1000 patients. Specifying 0 utilizes the full dataset for each optimization step, which is not typically advised due to the approximate nature of the gradient (better to have more frequent updates).
        verbose : bool, optional
            Indicates whether or not to print (to std out) the loss function values and error after every epoch. The default is True.


        Keyword Parameters
        ----------
        maxLearningRate: float, optional
            Specifies the maximum learning rate used during inference. Default is 0.04

        errorTol: float, optional
            Error tolerance in ELBO (computed on held out validation data) to determine convergence. Default is 1e-4.

        numParticles: int, optional
            Number of particles (ie random samples) used to approximate gradient. Default is 1. Computational cost increases linearly with value.

        maxEpochs: int, optional
            Maximum number of epochs (passes through training data) for inference. Note, because annealing and learning rate updates depend on maxEpochs, this offers a simple way to adjust the speed at which these values change.

        computeDevice: int or None, optional
            Specifies compute device for inference. Default is None, which instructs algorithm to use cpu. If integer is provided, then algorithm will be assigned to that integer valued gpu.

        numDataLoaders: int
            Specifies the number of threads used to process data and prepare for upload into the gpu. Note, due to the speed of gpu, inference can become limited by data transfer speed, hence the use of multiple DataLoaders to improve this bottleneck. Default is 0, meaning just the dedicated cpu performs data transfer.

        OneCycleParams: dict with keys 'pctCycleIncrease','initLRDivisionFactor','finalLRDivisionFactor'
            Parameters specifying the One-Cycle learning rate adjustment strategy, which helps to enable good anytime performance.
            pctCycleIncrease--fraction of inference epochs used for increasing learning rate. Default: 0.1
            initLRDivisionFactor--initial learning rate acheived by dividing maxLearningRate by this value. Default: 25.0
            finalLRDivisionFactor--final learning rate acheived by dividing maxLearningRate by this value. Default: 1e4


        KLAnnealingParams: dict with keys 'initialTemp','maxTemp','fractionalDuration','schedule'
            Parameters that define KL-Annealing strategy used during inference, important for avoiding local optima. Note, annealing is only used for computation of ELBO and gradients on training data. Validation data ELBO evaluation, used to monitor convergence, is performed at the maximum desired temperature (typically 1.0, equivalent to standard variational inference). Therefore, it is possible for the model to converge even when the temperature hasn't reached it's final value. It's possible that further cooling would find a better optimum, but this is highly unlikely in practice.
            initialTemp--initial temperature during inference. Default: 0.0
            maxTemp--final temperature obtained during inference. Default: 1.0 (standard variational inference)
            fractionalDuration--fraction of inference epochs used for annealing. Default is 0.25
            schedule--function used to change temperature during inference. Defualt is 'cosine'. Options: 'cosine','linear'

        'OpimizationStrategy': str, optional
            Specifies a strategry for optimization. Options include: 'Full','DecoderOnly','EncoderOnly'. Useful for debugging. Default is 'Full'


        Returns
        -------
        output : list
            List containing the following information: [loss function value of best model (computed on validation data),sequence of training loss values, sequence of validation loss values, error estimates across iterations (computed on validation data)].



        """


        ######### Parse Keyword Arguments #########
        allKeywordArgs = list(kwargs.keys())


        if 'maxLearningRate' in allKeywordArgs:
            maxLearningRate=kwargs['maxLearningRate']
        else:
            maxLearningRate=0.04


        if 'errorTol' in allKeywordArgs:
            errorTol=kwargs['errorTol']
        else:
            errorTol=1e-4

        if 'numParticles' in allKeywordArgs:
            numParticles=kwargs['numParticles']
        else:
            numParticles=1


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

        if 'OneCycleParams' in allKeywordArgs:
            OneCycleParams=kwargs['OneCycleParams']
            assert set(OneCycleParams.keys())==set(['pctCycleIncrease','initLRDivisionFactor','finalLRDivisionFactor']),"One-cycle LR scheduler requires the following parameters:'pctCycleIncrease','initLRDivisionFactor','finalLRDivisionFactor'"

        else:
            OneCycleParams={'pctCycleIncrease':0.1,'initLRDivisionFactor':25.0,'finalLRDivisionFactor':1e4}

        if 'KLAnnealingParams' in allKeywordArgs:
            KLAnnealingParams=kwargs['KLAnnealingParams']
            assert set(KLAnnealingParams.keys())==set(['initialTemp','maxTemp','fractionalDuration','schedule']),"KL Annealing Parameters must be dictionary with the following keys: 'initialTemp','maxTemp','fractionalDuration','schedule'"
        else:
            KLAnnealingParams={'initialTemp':0.0,'maxTemp':1.0,'fractionalDuration':0.25,'schedule': 'cosine'}

        if 'OpimizationStrategy' in allKeywordArgs:
            optimStrategy = kwargs['OpimizationStrategy']
            assert optimStrategy in ['Full','DecoderOnly','EncoderOnly'],"Available optimization strategies include: Full, DecoderOnly, EncoderOnly"
        else:
            optimStrategy='Full'


        pyro.clear_param_store()

        optimizer=Optimizer(self.model,self.sampler,optimizationParameters={'maxLearningRate': maxLearningRate,'maxEpochs': maxEpochs,'numParticles':numParticles},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},OneCycleParams=OneCycleParams,KLAnnealingParams=KLAnnealingParams)
        if batch_size > 0:
            output=optimizer.BatchTrain(batch_size,errorTol=errorTol,optimizationStrategy=optimStrategy,verbose=verbose)
        else:
            output=optimizer.FullDatasetTrain(errorTol=errorTol,optimizationStrategy=optimStrategy,verbose=verbose)
        print('Inference complete. Final Loss: {0:10.2f}'.format(output[0]))
        return output



    def ComputeEmbeddings(self,dataArrays=None,randomize=False,returnStdErrors=False):
        """
        Returns low dimnsional embeddings for the dataset. By default, perfoms
        operation on full testing and training dataset contained withinin sampler
        unless dataArrays are provided, in which case embeddings are computed
        for the provided arrays.


        Parameters
        ----------
        dataArray : 2-tuple of array-like data structures, corresponding to
        (incicidence array, [list of categorical covariates]), optional
            Can provide custom arrays not contained within ClinicalDatasetSampler
            in order to compute embeddings. The default value is None.
        randomize : bool, optional
            Determines whether to shuffle training/testing data prior to
            computation over data contained within ClinicalDatasetSampler.
            The default is False.
        returnStdErrors: bool, optional
            Indicates whether to return std errors for the embeddings in addition to the means.

        Returns
        -------

        Note: If std errors are included, then each of the below returned arrays will have 3-dimensions, wihth the third providing [mean,scale].

        numpy array
            training set embeddings.
        numpy array
            valdiation set embeddings.

        -- or --

        numpy array
            embeddings for dataArrays

        """

        if dataArrays is None:

            previousArrayType = self.sampler.returnArrays
            if self.sampler.returnArrays!='Torch':
                self.sampler.ChangeArrayType('Torch')

            trainingData=self.sampler.ReturnFullTrainingDataset(randomize=randomize)
            testingData=self.sampler.ReturnFullTestingDataset(randomize=randomize)

            train_embed = self.model.PredictLatentPhenotypes(trainingData[0],trainingData[1],returnScale=returnStdErrors)
            test_embed = self.model.PredictLatentPhenotypes(testingData[0],testingData[1],returnScale=returnStdErrors)

            if previousArrayType!='Torch':
                self.sampler.ChangeArrayType(previousArrayType)

            if returnStdErrors:

                return np.concatenate((train_embed[0].detach().numpy()[:,:,np.newaxis],train_embed[1].detach().numpy()[:,:,np.newaxis]), axis=-1),np.concatenate((test_embed[0].detach().numpy()[:,:,np.newaxis],test_embed[1].detach().numpy()[:,:,np.newaxis]),axis=-1)
            else:
                return train_embed.detach().numpy(),test_embed.detach().numpy()

        else:
            incidence_array = dataArrays[0]

            if torch.is_tensor(incidence_array) is False:
                incidence_array=self.sampler._torchWrapper(incidence_array)

            list_cov_arrays=[]
            for x in dataArrays[1]:
                if torch.is_tensor(x) is False:
                    x=self.sampler._torchWrapper(x)
                list_cov_arrays+=[x]

            embeddings=self.model.PredictLatentPhenotypes(incidence_array, list_cov_arrays,returnScale=returnStdErrors)

            if returnStdErrors:
                return np.concatenate((embeddings[0].detach().numpy()[:,:,np.newaxis],embeddings[1].detach().numpy()[:,:,np.newaxis]),axis=-1)
            else:
                return embeddings.detach().numpy()



    def ReturnComponents(self):
        """
        Returns the matrix of parameters defining f(Z), the function mapping the latent phenotypic space to the observed symptom risk values (sometimes called 'loadings' in factor analysis).
        This function is only available for linear f(Z).

        Returns
        -------
        numpy.array
            Matrix of latent phenotype loadings.

        """

        assert self.model.decoderType in ['Linear_Monotonic','Linear'],"Components only availabe for Linear models. vLPI fit using a non-linear mapping function."
        mapping_func_state_dict = self.model.decoder.state_dict()

        if self.model.decoderType == 'Linear_Monotonic':
            return torch.nn.functional.softplus(mapping_func_state_dict['linear_latent.log_scale_weight']).detach().numpy().T
        else:
            return mapping_func_state_dict['linear_latent.weight'].detach().numpy().T



    def ComputePerplexity(self,dataArrays=None, randomize=False):
        """
        Computes the per-datum perplexity (-1.0*ELBO) of the training/test data or the provided dataArrays.
        Can be thought of as a type of reconstruction error (sometimes also referred to as data "surprisal")

        Parameters
        ----------
        dataArray : 2-tuple of array-like data structures, corresponding to
        (incicidence array, [list of categorical covariates]), optional
            Can provide custom arrays not contained within ClinicalDatasetSampler
            in order to compute embeddings. The default value is None.
        randomize : bool, optional
            Determines whether to shuffle training/testing data prior to
            computation over data contained within ClinicalDatasetSampler.
            The default is False.

        Returns
        -------
        numpy.array
            Array of per-datum perplexity values.

        """
        if dataArrays is None:
            previousArrayType = self.sampler.returnArrays
            if self.sampler.returnArrays!='Torch':
                self.sampler.ChangeArrayType('Torch')

            trainingData=self.sampler.ReturnFullTrainingDataset(randomize=randomize)
            testingData=self.sampler.ReturnFullTestingDataset(randomize=randomize)

            train_elbo = self.model.ComputeELBOPerDatum(trainingData[0],trainingData[1])
            test_elbo = self.model.ComputeELBOPerDatum(testingData[0],testingData[1])

            if previousArrayType!='Torch':
                self.sampler.ChangeArrayType(previousArrayType)
            return -1.0*train_elbo.detach().numpy(),-1.0*test_elbo.detach().numpy()
        else:
            incidence_array = dataArrays[0]

            if torch.is_tensor(incidence_array) is False:
                incidence_array=self.sampler._torchWrapper(incidence_array)

            list_cov_arrays=[]
            for x in dataArrays[1]:
                if torch.is_tensor(x) is False:
                    x=self.sampler._torchWrapper(x)
                list_cov_arrays+=[x]
            elbo=self.model.ComputeELBOPerDatum(incidence_array, list_cov_arrays)
            return -1.0*elbo.detach().numpy()


    def _readModelFromFile(self,fName):
        with open(fName,'rb') as f:
            model_dict = torch.load(f,map_location='cpu')
        return model_dict


    def LoadModel(self,stored_model):
        """
        Loads previously fit model either from a dictionary (generated using PackageModel) or from a file path (with file constructed using PackageModel)

        Parameters
        ----------
        stored_model : either dict or str (file path)

        Returns
        -------
        None.

        """
        if not isinstance(stored_model,dict):
            assert isinstance(stored_model,str),"Expects file name if not provided with dictionary."
            stored_model = self._readModelFromFile(stored_model)
        assert set(stored_model.keys())==set(['model_state','meta_data']),"Model dictionary must contain the following elements: 'model_state','meta_data'"

        #first load meta data, make sure it aligns with structure of current vlpi instance

        self.model = VAE(stored_model['meta_data']['numObsTraits'], stored_model['meta_data']['numCatList'],stored_model['meta_data']['nLatentDim'],stored_model['meta_data']['latentPhenotypeMap'],**self.all_model_kwargs)
        self.model.LoadPriorState(stored_model)



    def PackageModel(self,fName=None):
        """
        Packages the current model and returns it as a python dictionary. Will optionally write this dictionary to disk using PyTorch.

        Parameters
        ----------
        fName : str, default None
            File path to save model to disk. The default is None, which means that only a model dictionary will be returned.

        Returns
        -------
        model_dict : dict
            Dictionary containing fitted model parameters in addition to general meta data.

        """
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
