#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:18:43 2019

@author: davidblair
"""
import copy
import torch
import pyro
import numpy as np
from vlpi.optim.Optimizer import Optimizer
from vlpi.data.ClinicalDataset import ClinicalDatasetSampler,ClinicalDataset
from vlpi.model.VAE import VAE
from vlpi.model.Classifier import Classifier as _Classifier
from sklearn.linear_model import LogisticRegression

class vLPI:


    def __init__(self,datasetSampler,nLatentDim,latentPhenotypeMap='Linear_Monotonic',**kwargs):
        """
        vLPI (variational-Latent Phenotype Inference) is a statistical model that maps a latent phenotypic space to some vector of observed (binary) traits through a function f(Z), where Z is the latent phenotypic space of interest. Z is assumed to be an isotropic multivariate gaussian distribution, which is equivalent to assuming independence among the latent components. Inference of the model is conducted using a variational approximation of the model marginal model likelihood, which is optimized through stochastic gradient descent. To allow inference to scale to millions of patients and enable subsequent portability, latent phenotype inference is amortized using a non-linear neural network.

        In addition, different medical record datasets are often encoded using slightly or even significantly different terminologies. For example, the UK Biobank is encoded in ICD10, while most American medical record systems have implemented using ICD10-CM (or ICD9). To translate latent spaces across datasets, the class also contains functions (specifically ConstructTranslationEncoder) that enables the training of latent-space encoding function that uses a different terminology than the one used to train the initial model. Training this translation encoder requires a dataset in which both terminologies have been used, which is the case for terminologies like ICD9 and ICD10 or terminologies that are more or less strict subsets/supersets of one another (ICD10 and ICD10-CM).

        Parameters
        ----------
        datasetSampler : ClinicalDatasetSampler
            Class that contains the ClinicalData and a mechanism forn (efficiently) generating random samples.
        nLatentDim : int
            Number of latent phenotypes to include into the model.
        latentPhenotypeMap : str, optional
            Function f(Z) mapping latent to observed phenotypes. Must be one of the following ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic']. The default is 'Linear_Monotonic'. Initial testing has only been performed using this function.
        **kwargs: These are arguments passed to the class VAE, which is the modeling class (writtent in PyTorch and pyro) used for inference.

            linkFunction: Function used to map risk-scores produced by f(Z) to the probability scale [0.0,1.0]. Must be in ['Logit','Probit']. Note, 'Logit' is considerably faster.

            dropLinearCovariateColumn: boolean value indicated whether to drop one category from each covariate included into the model. Aids in identifiability. Defaults to True.

            encoderNetworkHyperparameters: dictionary that speficies hyperparameters of the encoder neural networks. Default is a 2-layer MLP with 64 hidden nodes per layer. Dropout rate 0.0 with Batch-Normalization enabled. Larger networks are more difficult to train, while smaller networks are less expressive and may result in poorer approximations of the variational posterior. Based on testing, allowing for dropout appears to limit the models ability to detect rare events. Default: {'n_layers' : 2, 'n_hidden' : 64, 'dropout_rate': 0.0, 'use_batch_norm':True}

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
            Size of dataset batches for inference. The default is 1000 patients. Specifying 0 utilizes the full dataset for each optimization step, which is not typically advised due to the approximate nature of the gradient.
        verbose : bool, optional
            Indicates whether or not to print (to std out) the loss function values and error after every epoch. The default is True.
        **kwargs : TYPE
            DESCRIPTION.

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


        if 'finalErrorTol' in allKeywordArgs:
            finalErrorTol=kwargs['finalErrorTol']
        else:
            finalErrorTol=1e-4

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
            output=optimizer.BatchTrain(batch_size,errorTol=finalErrorTol,optimizationStrategy=optimStrategy,verbose=verbose)
        else:
            output=optimizer.FullDatasetTrain(errorTol=finalErrorTol,optimizationStrategy=optimStrategy,verbose=verbose)
        print('Inference complete. Final Loss: {0:10.2f}'.format(output[0]))
        return output



    def ComputeEmbeddings(self,dataArrays=None,randomize=False,returnStdErrors=False):
        """
        Returns low dimnsional embeddings for dataset. By default, perfoms
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
            training set embeddings.

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
                incidence_array=sampler._torchWrapper(incidence_array)

            list_cov_arrays=[]
            for x in dataArrays[1]:
                if torch.is_tensor(x) is False:
                    x=sampler._torchWrapper(x)
                list_cov_arrays+=[x]

            embeddings=self.model.PredictLatentPhenotypes(incidence_array, list_cov_arrays,returnScale=returnStdErrors)

            if returnStdErrors:
                return np.concatenate((embeddings[0].detach().numpy()[:,:,np.newaxis],embeddings[1].detach().numpy()[:,:,np.newaxis]),axis=-1)
            else:
                return embeddings.detach().numpy()



    def ComputeMorbidityScores(self,dataArrays=None,centroid=None):

        if dataArrays is None:
            trainEmbeddings,testEmbeddings = self.ComputeEmbeddings()
            if centroid is None:
                centroid = np.mean(trainEmbeddings,axis=0)

            vec_direcs_train = trainEmbeddings-centroid
            vec_direcs_train[vec_direcs_train<0.0]=0.0

            vec_direcs_test = testEmbeddings-centroid
            vec_direcs_test[vec_direcs_test<0.0]=0.0
            return np.sqrt(np.sum(vec_direcs_train**2.0,axis=1)),np.sqrt(np.sum(vec_direcs_test**2.0,axis=1))

        else:
            incidence_array = dataArrays[0]

            if torch.is_tensor(incidence_array) is False:
                incidence_array=sampler._torchWrapper(incidence_array)

            list_cov_arrays=[]
            for x in dataArrays[1]:
                if torch.is_tensor(x) is False:
                    x=sampler._torchWrapper(x)
                list_cov_arrays+=[x]

            embeddings=self.model.PredictLatentPhenotypes(incidence_array, list_cov_arrays).detach().numpy()
            if centroid is None:
                trainEmbeddings,testEmbeddings = self.ComputeEmbeddings()
                centroid = np.mean(trainEmbeddings,axis=0)

            vec_direcs= embeddings-centroid
            vec_direcs[vec_direcs<0.0]=0.0
            return np.sqrt(np.sum(vec_direcs**2.0,axis=1))


    def ReturnComponents(self):

        assert self.model.decoderType in ['Linear_Monotonic','Linear'],"Components only availabe for Linear models. vLPI fit using a non-linear mapping function."
        mapping_func_state_dict = self.model.decoder.state_dict()

        if self.model.decoderType == 'Linear_Monotonic':
            return np.exp(mapping_func_state_dict['linear_latent.log_scale_weight'].detach().numpy()).T
        else:
            return mapping_func_state_dict['linear_latent.weight'].detach().numpy().T



    def ComputePerplexity(self,dataArrays=None):
        if dataArrays is None:
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
        else:
            incidence_array = dataArrays[0]

            if torch.is_tensor(incidence_array) is False:
                incidence_array=sampler._torchWrapper(incidence_array)

            list_cov_arrays=[]
            for x in dataArrays[1]:
                if torch.is_tensor(x) is False:
                    x=sampler._torchWrapper(x)
                list_cov_arrays+=[x]
            elbo=self.model.ComputeELBOPerDatum(incidence_array, list_cov_arrays)
            return -1.0*elbo.detach().numpy()

        
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


class Classifier:

    def __init__(self,datasetSampler,observedPhenotypeMap='Linear',**kwargs):
        """
        Python class that implements the supervised learning algorithms on clinical data.

        By default, FitModel trains the linear/non-linear classifier

        Use ElasticNet to access logistic regression in sklearn.

        """


        self.sampler = datasetSampler
        assert self.sampler.isConditioned, "ClinicalDatasetSampler must be conditioned on some diagnosis/laber of interest in order to use vlpi.Classifier!"

        assert observedPhenotypeMap in ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'], "Currently supported observed-to-label phenotype maps include: 'Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'"


        self.all_model_kwargs = kwargs

        n_cat_list = [len(self.sampler.currentClinicalDataset.catCovConversionDicts[x]) for x in self.sampler.includedCovariates]

        assert observedPhenotypeMap in ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'], "Currently supported latent-to-observed phenotype maps include: 'Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'"


        self.model = _Classifier(self.sampler.currentClinicalDataset.numDxCodes, n_cat_list,observedPhenotypeMap,**kwargs)

    def ElasticNet(self,verbose=True,**kwargs):

        allKeywordArgs = list(kwargs.keys())

        if 'errorTol' in allKeywordArgs:
            errorTol=kwargs['errorTol']
        else:
            errorTol = 1e-4

        if 'verbose' in allKeywordArgs:
            verbose=kwargs['verbose']
        else:
            verbose = True

        if 'maxFittingIterations' in allKeywordArgs:
            maxFittingIterations=kwargs['maxFittingIterations']
        else:
            maxFittingIterations=400

        if 'penaltyParam' in allKeywordArgs:
            penaltyParam = kwargs['penaltyParam']
        else:
            penaltyParam=1.0

        if 'L1Ratio' in allKeywordArgs:
            L1Ratio = kwargs['L1Ratio']
        else:
            L1Ratio=0.5


        previousArrayType = self.sampler.returnArrays
        if self.sampler.returnArrays!='Sparse':
            self.sampler.ChangeArrayType('Sparse')
        sparseTrainingData=self.sampler.ReturnFullTrainingDataset(randomize=False)
        sparseTestingData=self.sampler.ReturnFullTestingDataset(randomize=False)

        X_data_train = self.sampler.CollapseDataArrays(sparseTrainingData[0],sparseTrainingData[1],drop_column=True)
        X_data_test = self.sampler.CollapseDataArrays(sparseTestingData[0],sparseTestingData[1],drop_column=True)

        diseasePrevalence = self.sampler.currentClinicalDataset.data['has_'+self.sampler.conditionSamplingOnDx[0]].sum()/self.sampler.currentClinicalDataset.numPatients

        linMod = LogisticRegression(penalty='elasticnet',tol=errorTol,verbose=verbose,fit_intercept=True,max_iter=maxFittingIterations,solver='saga',warm_start=True,l1_ratio=L1Ratio,C=penaltyParam)
        linMod.intercept_ = np.array([np.log(diseasePrevalence)-np.log(1.0-diseasePrevalence)])
        linMod.coef_ = np.zeros((1,self.model.numObsTraits+self.model.numCovParam))
        linMod=linMod.fit(X_data_train,sparseTrainingData[2].toarray().ravel())

        pred_prob=linMod.predict_proba(X_data_test)

        if previousArrayType!='Sparse':
            self.sampler.ChangeArrayType(previousArrayType)

        return {'Model':linMod,'Prediction Scores':pred_prob[:,1]}


    def FitModel(self,batch_size=0,verbose=True,**kwargs):


        ######### Parse Keyword Arguments #########
        allKeywordArgs = list(kwargs.keys())


        if 'maxLearningRate' in allKeywordArgs:
            maxLearningRate=kwargs['maxLearningRate']
        else:
            maxLearningRate=0.04

        if 'errorTol' in allKeywordArgs:
            errorTol=kwargs['errorTol']
        else:
            errorTol = 1e-4


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


        pyro.clear_param_store()
        optimizer=Optimizer(self.model,self.sampler,optimizationParameters={'maxLearningRate': maxLearningRate,'maxEpochs': maxEpochs,'numParticles':numParticles},computeConfiguration={'device':computeDevice,'numDataLoaders':numDataLoaders},OneCycleParams=OneCycleParams)

        if batch_size > 0:
            output=optimizer.BatchTrain(batch_size,errorTol=errorTol,optimizationStrategy='Full',verbose=verbose)
        else:
            output=optimizer.FullDatasetTrain(errorTol=errorTol,optimizationStrategy='Full',verbose=verbose)
        print('Inference complete. Final ELBO:{0:10.2f}'.format(-1.0*output[0]))
        return -1.0*output[0]

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

        self.model = _Classifier(stored_model['meta_data']['numObsTraits'], stored_model['meta_data']['numCatList'],stored_model['meta_data']['observedPhenotypeMap'],**self.all_model_kwargs)
        self.model.LoadPriorState(stored_model)



    def PackageModel(self,fName=None):
        model_dict = self.model.PackageCurrentState()
        model_dict['meta_data']={}
        model_dict['meta_data']['numObsTraits']=self.model.numObsTraits
        model_dict['meta_data']['numCatList']=self.model.numCatList
        model_dict['meta_data']['observedPhenotypeMap']=self.model.decoderType

        if fName is not None:
            with open(fName,'wb') as f:
                torch.save(model_dict,f)
        return model_dict

    def PredictTestLabels(self,dataArrays=None):
        if dataArrays is None:
            previousArrayType = self.sampler.returnArrays
            if self.sampler.returnArrays!='Torch':
                self.sampler.ChangeArrayType('Torch')

            testingData=self.sampler.ReturnFullTestingDataset(randomize=False)

            test_pred= self.model.PredictLabels(testingData[0],testingData[1])

            if previousArrayType!='Torch':
                self.sampler.ChangeArrayType(previousArrayType)
            return test_pred.detach().numpy()
        else:
            incidence_array = dataArrays[0]

            if torch.is_tensor(incidence_array) is False:
                incidence_array=sampler._torchWrapper(incidence_array)

            list_cov_arrays=[]
            for x in dataArrays[1]:
                if torch.is_tensor(x) is False:
                    x=sampler._torchWrapper(x)
                list_cov_arrays+=[x]
            pred=self.model.PredictLabels(incidence_array,list_cov_arrays)
            return pred.detach().numpy()



if __name__=='__main__':

    from vlpi.data.ClinicalDataSimulator import ClinicalDataSimulator

    pyro.clear_param_store()

    numSamples = 10000
    numTotalTraits=100
    numAssociatedTraits=20
    nLatentSimDim=4
    nLatentFitDim=4
    mappingFunction='Linear_Monotonic'

    numCovPerClass = [2,3,10]
    covNames = ['A','B','C']
    numCovPerClass = []
    covNames = []


    simulator = ClinicalDataSimulator(numTotalTraits,nLatentSimDim,numCovPerClass,interceptPriors=[-3.0,1.5])
    simData=simulator.GenerateClinicalData(numSamples)
    labelData=simulator.GenerateTargetDx(simData['latent_phenotypes'])

    clinData = ClinicalDataset()
    altClinData = ClinicalDataset()


    tmp_index = np.arange(numTotalTraits)
    possible_dis_list = np.array(list(clinData.dxCodeToDataIndexMap.keys()))
    np.random.shuffle(tmp_index)

    disList =possible_dis_list[tmp_index[0:numAssociatedTraits]]

    clinData.IncludeOnly(disList)
    clinData.LoadFromArrays(simData['incidence_data'][:,tmp_index[0:numAssociatedTraits]],simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')
    sampler = ClinicalDatasetSampler(clinData,0.75,returnArrays='Torch')


    np.random.shuffle(tmp_index)
    disList = possible_dis_list[tmp_index[0:numAssociatedTraits]]
    altClinData.IncludeOnly(disList)
    altClinData.LoadFromArrays(simData['incidence_data'][:,tmp_index[0:numAssociatedTraits]],simData['covariate_data'],covNames,catCovDicts=None, arrayType = 'Torch')



    val_sampler = sampler.GenerateValidationSampler(0.2)
    vlpi_model = vLPI(val_sampler,nLatentFitDim)
    output=vlpi_model.FitModel(1000,finalErrorTol=1e-4,max_epochs=500)




    # mean_2,scale_2=translation_encoder_2(data[3],data[1])

    # # new_vlpi_model = vLPI(val_sampler,nLatentFitDim)
    # # output=new_vlpi_model.FitModel(1000,scoreInitializationParameters=(patient_index,embedding_means,embedding_stds),finalErrorTol=1e-4,maxLearningRate=0.04,maxEpochs=2000,KLAnnealingParams={'initialTemp':1.0,'maxTemp':1.0,'fractionalDuration':0.25,'schedule': 'cosine'})

    # new_train_embed,new_test_embed=new_vlpi_model.ComputeEmbeddings(dataArrays=val_sampler.ReturnFullTrainingDataset(randomize=False),returnStdErrors=True)









#    training_classifier=Classifier(val_sampler,observedPhenotypeMap='Nonlinear')
#    testing_classifer = Classifier(sampler,observedPhenotypeMap='Nonlinear')
#
#    output_NN=training_classifier.FitModel(batch_size=5000,maxLearningRate=0.05)
#    model = training_classifier.PackageModel()
#    testing_classifer.LoadModel(model)
#
#    output_eNet = testing_classifer.ElasticNet(penaltyParam=1.0)
#
#    test_pred = testing_classifer.PredictTestLabels()
#    pr = precision_recall_curve(labelData['label_dx_data'].numpy()[np.concatenate(sampler.testDataIndex)],test_pred)
#    plt.step(pr[1],pr[0])
#    pr = precision_recall_curve(labelData['label_dx_data'].numpy()[np.concatenate(sampler.testDataIndex)],output_eNet['Prediction Scores'])
#    plt.step(pr[1],pr[0])
#    plt.figure()
#    plt.plot(test_pred,output_eNet['Prediction Scores'],'o')
#
