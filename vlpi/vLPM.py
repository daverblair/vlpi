#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from typing import Iterable
from collections import OrderedDict
import pickle
import os
from vlpi.Decoders import LinearDecoder,LinearDecoder_Monotonic,NonlinearMLPDecoder,NonlinearMLPDecoder_Monotonic
from vlpi.Encoders import MeanScaleEncoder
from vlpi.utils import random_catcov,infer_beta_from_CI

class vLPM(nn.Module):

    """

    This is a statistical model that maps a latent phenotypic space to some vector of observed (binary) phenotypes through a function f(z), where z is the latent phenotypic space of interest. This model can also use information provided in the form of labels marking the extreme end of the latent phenotypic spectrum (ie a monogenic disease dx) to improve inference, similar to supervised learning. However, it does not treat the lablels as fixed and known. Instead, the labels are used as additional information for model fitting, and the true diagnostic status of each subject is treated as an unknown latent variable, which is marginalized out of the model during inference.

    More formally, let \mathbf{Y} denote a vector of binary random variables and \mathbf{z} denote a latent phenotyptic space where \mathbf{z}~MVN(0, I). P(Y_{i}) = \Phi(\alpha+f(A)+f(z)), where is A is a matrix of possible confounding factors \Phi(X) is the CDF function for the standard normal applied element-wise.


    Model inference is conducted through variational inference using SGD and amoritization using a non-linear NN where applicable.

    """

    def _setEvalMode(self):
        self.encoder.eval()
        self.decoder.eval()

    def _setTrainMode(self):
        self.encoder.train()
        self.decoder.train()

    def __init__(self, numObsTraits:int, numCatList:Iterable[int],numLatentDx:int,nLatentDim:int,mappingFunction:str,dxCodeSensitivity_95CI:Iterable[float] = [0.1,0.9],dxCodeSpecificity_95CI:Iterable[float] = [0.99,0.999],biasPrior:Iterable[float]=[0.0,5.0],**kwargs):


        """
        numObsTraits-->number of traits geneterated from latent phenotype
        numCatList-->List of the number of categories for each categorical covariate; total # of categorical covariates = len(numCatList).
        numLatentDx-->number of labels for the latent phenotype included in the data. Modeled as the noisy output of a latent true disease status.
        nLatentDim-->number of dimensions composing the latent phenotype. Fixed at 1 if numLatentDx > 0.
        mappingFunction-->type of function mapping latent phenotype space to observed traits. Options include: Linear, Linear_Monotonic, Nonlinear, Nonlinear_Monotonic.
        dxCodeSensitivity_95CI-->prior information for latent disease dx code sensitivity, expressed as a 95% credible interval for a Beta distributed RV.
        dxCodeSpecificity_95CI-->prior information for latent disease dx code specificity, expressed as a 95% credible interval for a Beta distributed RV.
        **kwargs: There are multiple keyword arguments available as well. These handle many of the hyperparameters and more esoteric modelling options. Information regarding each is provided below
        """


        super(vLPM,self).__init__()
        self.numObsTraits=numObsTraits
        self.numCatList=numCatList
        self.numLatentDx=numLatentDx
        if self.numLatentDx>0:
            assert nLatentDim==1,"Model currently supports only a single latent phenotype dimension when latent disease diagnoses are provided."
        self.nLatentDim = nLatentDim
        assert mappingFunction in ['Linear','Linear_Monotonic','Nonlinear','Nonlinear_Monotonic'], "Currently supported model types for vLPM: Linear, Linear_Monotonic, Nonlinear,Nonlinear_Monotonic."
        self.mappingFunction=mappingFunction
        self.dxCodeSensitivityPrior = torch.tensor(infer_beta_from_CI(dxCodeSensitivity_95CI),dtype=torch.float)
        self.dxCodeSpecificityPrior = torch.tensor(infer_beta_from_CI(dxCodeSpecificity_95CI),dtype=torch.float)
        self.biasPrior = torch.tensor(biasPrior)


        #kwargs
        allKeywordArgs = list(kwargs.keys())

        if 'dropLinearCovariateColumn' not in allKeywordArgs:
            """
            Specifies model to drop one category from each covariate. Defaults to True.
            """
            self.dropLinearCovariateColumn=True
        else:
            self.dropLinearCovariateColumn=kwargs['dropLinearCovariateColumn']
            assert isinstance(self.dropLinearCovariateColumn,bool),"dropLinearCovariateColumn expects boolean value"

        if 'coupleCovariates' not in allKeywordArgs:
            """
            Specifies whether to couple covariate variables into Nonlinear NN mapping function
            """
            self.coupleCovariates=False
        else:
            self.coupleCovariates=kwargs['coupleCovariates']
            assert isinstance(self.coupleCovariates,bool),"coupleCovariates expects boolean value"
            assert mappingFunction=='Nonlinear',"Covariate coupling only available with Nonlinear mapping function."

        if 'encoderHyperparameters' not in allKeywordArgs:
            self.encoderHyperparameters={'n_layers' : 2, 'n_hidden' : 128, 'dropout_rate': 0.1, 'use_batch_norm':True}

        else:
            self.encoderHyperparameters = kwargs['encoderHyperparameters']
            assert isinstance(self.encoderHyperparameters,dict),"Expects dictionary of encoder hyperparameters"
            assert set(self.encoderHyperparameters.keys())==set(['n_layers','n_hidden','dropout_rate','use_batch_norm']),"Encoder hyperparameters must include: 'n_layers','n_hidden','dropout_rate','use_batch_norm'"

        if 'decoderHyperparameters' not in allKeywordArgs:
            self.decoderHyperparameters={'n_layers' : 2, 'n_hidden' : 128, 'dropout_rate': 0.1, 'use_batch_norm':True}
        else:
            self.decoderHyperparameters=kwargs['decoderHyperparameters']
            assert isinstance(self.decoderHyperparameters,dict),"Expects dictionary of encoder hyperparameters"
            assert set(self.decoderHyperparameters.keys())==set(['n_layers','n_hidden','dropout_rate','use_batch_norm']),"Encoder hyperparameters must include: 'n_layers','n_hidden','dropout_rate','use_batch_norm'"


        self.encoder=MeanScaleEncoder(self.numObsTraits+self.numLatentDx, self.nLatentDim, n_cat_list= self.numCatList, **self.encoderHyperparameters)

        if self.mappingFunction=='Linear':
            self.decoder = LinearDecoder(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn)
        elif self.mappingFunction=='Linear_Monotonic':
            self.decoder=LinearDecoder_Monotonic(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn)
        elif self.mappingFunction=='Nonlinear':
            self.decoder=NonlinearMLPDecoder(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn,self.coupleCovariates,**self.decoderHyperparameters)
        elif self.mappingFunction=='Nonlinear_Monotonic':
            self.decoder=NonlinearMLPDecoder_Monotonic(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn,**self.decoderHyperparameters)

        self.posteriorParmDict = {}

        self.posteriorParmDict['biasPosteriorMean'] = pyro.param("biasPosteriorMean",torch.zeros(self.numObsTraits))
        self.posteriorParmDict['biasPosteriorScale'] = pyro.param("biasPosteriorScale",torch.ones(self.numObsTraits),constraint=torch.distributions.constraints.positive)
        if self.numLatentDx > 0:
            self.posteriorParmDict['dxCodeSensitivityPosterior'] = pyro.param("dxCodeSensitivityPosterior",torch.ones(2,self.numLatentDx),constraint=torch.distributions.constraints.positive)
            self.posteriorParmDict['dxCodeSpecificityPosterior'] = pyro.param("dxCodeSpecificityPosterior",torch.ones(2,self.numLatentDx),constraint=torch.distributions.constraints.positive)


        self._setEvalMode()

    def SimuateData(self,numSamples):
        """
        Draws numSamples of samples from model.
        """

        cat_cov_list = []
        for n_cat in self.numCatList:
            cat_cov_list+=[random_catcov(n_cat,numSamples)]
        paramDict = {}

        bias = dist.Normal(self.biasPrior[0]*torch.ones(self.numObsTraits),self.biasPrior[1]*torch.ones(self.numObsTraits)).sample()

        paramDict['bias'] = bias

        if self.numLatentDx>0:
            dxCodeSensitivity = dist.Beta(self.dxCodeSensitivityPrior[0],self.dxCodeSensitivityPrior[1]).sample((1,self.numLatentDx))
            dxCodeSpecificity = dist.Beta(self.dxCodeSpecificityPrior[0],self.dxCodeSpecificityPrior[1]).sample((1,self.numLatentDx))
            paramDict['sensitivity']=dxCodeSensitivity
            paramDict['specificity']=dxCodeSpecificity


        latentPhenotype=dist.Normal(torch.zeros(numSamples,self.nLatentDim),torch.ones(numSamples,self.nLatentDim)).sample()
        mean_obs_dis = self.decoder(latentPhenotype,*cat_cov_list)+bias

        gen_obs_data =dist.Bernoulli(dist.Normal(0.0,1.0).cdf(mean_obs_dis)).sample()

        if self.numLatentDx>0:
            has_latent_dis = dist.Bernoulli(dist.Normal(0.0,1.0).cdf(latentPhenotype)).sample()
            latent_dx_prob = dxCodeSensitivity*has_latent_dis+(1.0-has_latent_dis)*(1.0-dxCodeSpecificity)
            latent_dx=dist.Bernoulli(latent_dx_prob).sample()
        if self.numLatentDx>0:
            return cat_cov_list,gen_obs_data,latent_dx,has_latent_dis,latentPhenotype,paramDict
        else:
            return cat_cov_list,gen_obs_data,latentPhenotype,paramDict

    def model(self, obs_data=None, cat_cov_list=None, latent_dx=None):
        assert obs_data is not None,"Model requires input data. To simulate random samples, use SimulateData function."
        numSamples=obs_data.shape[0]
        if self.numLatentDx > 0:
            assert latent_dx is not None,"Model requires input of latent dx data if marked as included."
            assert latent_dx.shape[0]==obs_data.shape[0],"Dimensions of latent and observed dx data must match."
            data = torch.cat((obs_data,latent_dx),dim=-1)
        else:
            data=obs_data

        pyro.module("decoder", self.decoder,update_module_params=True)
        bias = pyro.sample("bias",dist.Normal(self.biasPrior[0]*torch.ones(self.numObsTraits),self.biasPrior[1]*torch.ones(self.numObsTraits)).to_event(1))

        if self.numLatentDx > 0:
            dxCodeSensitivity = pyro.sample("dxCodeSensitivity",dist.Beta(self.dxCodeSensitivityPrior[0]*torch.ones((1,self.numLatentDx)),self.dxCodeSensitivityPrior[1]*torch.ones((1,self.numLatentDx))))
            dxCodeSpecificity = pyro.sample("dxCodeSpecificity",dist.Beta(self.dxCodeSpecificityPrior[0]*torch.ones((1,self.numLatentDx)),self.dxCodeSpecificityPrior[1]*torch.ones((1,self.numLatentDx))))

        latentPhenotype=pyro.sample("latentPhenotype",dist.Normal(torch.zeros(numSamples,self.nLatentDim),torch.ones(numSamples,self.nLatentDim)).to_event(1))

        liabilities = self.decoder(latentPhenotype,*cat_cov_list)+bias
        if self.numLatentDx > 0:
            has_latent_dis_prob = dist.Normal(0.0,1.0).cdf(latentPhenotype)
            latent_dx_prob = dxCodeSensitivity*has_latent_dis_prob+(1.0-has_latent_dis_prob)*(1.0-dxCodeSpecificity)
            mean_dx_rates = torch.cat((dist.Normal(0.0,1.0).cdf(liabilities),latent_dx_prob),dim=-1)
        else:
            mean_dx_rates=dist.Normal(0.0,1.0).cdf(liabilities)
        pyro.sample("obsTraitIncidence",dist.Bernoulli(mean_dx_rates).to_event(1),obs=data)

    def guide(self,obs_data=None, cat_cov_list=None,latent_dx=None):
        assert obs_data is not None,"Guide requires input data. To simulate random samples, use SimulateData function."
        if self.numLatentDx>0:
            assert latent_dx is not None,"Model requires input of latent dx data if marked as included."
            assert latent_dx.shape[0]==obs_data.shape[0],"Dimensions of latent and observed dx data must match."

        self.posteriorParmDict['biasPosteriorMean'] = pyro.param("biasPosteriorMean",torch.zeros(self.numObsTraits))
        self.posteriorParmDict['biasPosteriorScale'] = pyro.param("biasPosteriorScale",torch.ones(self.numObsTraits),constraint=torch.distributions.constraints.positive)
        pyro.module("encoder", self.encoder,update_module_params=True)
        pyro.sample("bias",dist.Normal(self.posteriorParmDict['biasPosteriorMean'] ,self.posteriorParmDict['biasPosteriorScale']).to_event(1))

        if self.numLatentDx > 0:
            self.posteriorParmDict['dxCodeSensitivityPosterior'] = pyro.param("dxCodeSensitivityPosterior",torch.ones((2,self.numLatentDx)),constraint=torch.distributions.constraints.positive)
            self.posteriorParmDict['dxCodeSpecificityPosterior'] = pyro.param("dxCodeSpecificityPosterior",torch.ones((2,self.numLatentDx)),constraint=torch.distributions.constraints.positive)
            pyro.sample("dxCodeSensitivity",dist.Beta(self.posteriorParmDict['dxCodeSensitivityPosterior'][0,:],self.posteriorParmDict['dxCodeSensitivityPosterior'][1,:]))
            pyro.sample("dxCodeSpecificity",dist.Beta(self.posteriorParmDict['dxCodeSpecificityPosterior'][0,:],self.posteriorParmDict['dxCodeSpecificityPosterior'][1,:]))

        if self.numLatentDx > 0:
            z_mean,z_std = self.encoder(torch.cat((obs_data,latent_dx),dim=-1),*cat_cov_list)
        else:
            z_mean,z_std = self.encoder(obs_data,*cat_cov_list)
        pyro.sample("latentPhenotype", dist.Normal(z_mean, z_std).to_event(1))



    def PredictTrueDisStatus(self,obs_dis_array,cat_cov_list,latent_dx_array=None,useNaive=False,nMonteCarlo = 1000):
        """
        Predicts the probability that patient has a particular latent disease given the model parameters.
        In the case where the latent disease diagnosis status is unknown (numLatentDx==0),
        this amounts to computing the latent phenotype expectation followed by the CDF for a standard Normal.

        When the latent disease diagnosis status is known, the posterior predictive distribution is more complicated
        and requires the computation of a 3-d integral without an analytical solution. In theory, this integral can be computed
        using the monte carlo method (ie drawing samples from the parameter posteriors), but simply replacing the parameters
        with their expectations (useNaive==True) performs quite well in practice.

        """
        if self.numLatentDx==0:
            z_mean,z_scale = self.encoder(obs_dis_array,*cat_cov_list)
            return dist.Normal(0.0,1.0).cdf(z_mean),z_mean

        assert self.posteriorParmDict['dxCodeSpecificityPosterior'] is not None, "Model infernece not yet performed. Unable to use function."
        assert latent_dx_array is not None,"Must provide latent dis dx vector if marked within the model."
        z_mean,z_scale = self.encoder(torch.cat((obs_dis_array,latent_dx_array),dim=-1),*cat_cov_list)


        sensPostParams = self.posteriorParmDict['dxCodeSensitivityPosterior']
        specPostParams = self.posteriorParmDict['dxCodeSpecificityPosterior']


        has_dis_prior = lambda x: dist.Normal(0,1).cdf(x)
        sensPostDist=dist.Beta(sensPostParams[0],sensPostParams[1])
        specPostDist=dist.Beta(specPostParams[0],specPostParams[1])

        postPredHasDis=[]
        for i in range(latent_dx_array.shape[0]):
            ldx = latent_dx_array[i]
            post_pred_dis = lambda x,y,z: (((x*ldx)+((1.0-x)*(1-ldx))).prod(dim=1)*has_dis_prior(z))/((((x*ldx)+((1.0-x)*(1-ldx))).prod(dim=1)*has_dis_prior(z))+((((1.0-y)*ldx)+y*(1-ldx)).prod(dim=1)*(1.0-has_dis_prior(z))))
            if useNaive:
                expSens,expSpec,expLP = sensPostParams[0:1,:]/sensPostParams.sum(dim=0),specPostParams[0:1,:]/specPostParams.sum(dim=0),z_mean[i,0]
                postPredHasDis+=[[post_pred_dis(expSens,expSpec,expLP)]]
            else:
                random_joint_samps = lambda n: (sensPostDist.sample((n,)),specPostDist.sample((n,)),dist.Normal(z_mean[i,0],z_scale[i,0]).sample((n,)))
                postPredHasDis+=[[post_pred_dis(*random_joint_samps(nMonteCarlo)).sum()/nMonteCarlo]]
        return torch.tensor(postPredHasDis),z_mean

    def ReturnPosteriorParameters(self):
        newDict=OrderedDict()
        for key,value in self.posteriorParmDict.items():
            newDict[key]=value.detach().clone()
        return newDict

    def WriteToFile(self,fDirec):
        """
        Writes current state of model (hyperparameters, encoder, decoder, posterior parameters)
        into four files into a directory named fDirec:
        1) mInfo: holds general model info (numObsTraits,nLatentDim,etc)
        2) encoder: holds encoder params
        3) decoder: holds decoder params
        4) mParams: holds pyro.param state


        """
        fDirec=fDirec.strip('/')
        os.mkdir(fDirec)
        assert self.posteriorParmDict['biasPosteriorMean'] is  not None, "Model infernece not yet performed. Unable to use function."
        currentModel = OrderedDict()


        currentModel['numObsTraits']=self.numObsTraits
        currentModel['numCatList']=self.numCatList
        currentModel['numLatentDx']=self.numLatentDx
        currentModel['nLatentDim']=self.nLatentDim
        currentModel['mappingFunction']=self.mappingFunction
        currentModel['dxCodeSensitivityPrior']=self.dxCodeSensitivityPrior
        currentModel['dxCodeSpecificityPrior']=self.dxCodeSpecificityPrior
        currentModel['biasPrior']=self.biasPrior
        currentModel['dropLinearCovariateColumn']=self.dropLinearCovariateColumn
        currentModel['coupleCovariates']=self.coupleCovariates
        currentModel['encoderHyperparameters']=self.encoderHyperparameters
        currentModel['decoderHyperparameters']=self.decoderHyperparameters
        currentModel['posteriorParmDict']=self.posteriorParmDict

        with open(fDirec+'/mInfo.pth', 'wb') as f:
            pickle.dump(currentModel,f)

        torch.save(self.encoder.state_dict(), fDirec+'/encoder.pth')
        torch.save(self.decoder.state_dict(), fDirec+'/decoder.pth')
        pStore = pyro.get_param_store()
        pStore.save(fDirec+'/mParams.pth')



    def LoadParametersFromFile(self,fDirec):
        """
        Loads model state from file. Note: function performs high-level check to
        make sure that model specifications are globally consisent (same numObsTraits,
        numLatentDx,numCatList,mappingFunction). However, it does not strictly
        check that prior distributions,encoder/decoder hyperparameters, etc strictly match,
        as this level of matching is typically undesirable/unnecessary for quickly loading
        models. Instead, it overwrites this information.

        """
        fDirec=fDirec.strip('/')
        with open(fDirec+'/mInfo.pth', 'rb') as f:
            currentModel = pickle.load(f)

        assert currentModel['numObsTraits']==self.numObsTraits, "numObsTraits does not match model saved in file"
        assert currentModel['numCatList']==self.numCatList, "numCatList does not match model saved in file"
        assert currentModel['numLatentDx']==self.numLatentDx,"numLatentDx does not match model saved in file"
        assert currentModel['nLatentDim']==self.nLatentDim,"nLatentDim does not match model saved in file"
        assert currentModel['mappingFunction']==self.mappingFunction,"mappingFunction does not match model saved in file"

        #all checks out, now load the data
        self.dxCodeSensitivityPrior = currentModel['dxCodeSensitivityPrior']
        self.dxCodeSpecificityPrior = currentModel['dxCodeSpecificityPrior']
        self.biasPrior = currentModel['biasPrior']
        self.dropLinearCovariateColumn = currentModel['dropLinearCovariateColumn']
        self.coupleCovariates = currentModel['coupleCovariates']
        self.encoderHyperparameters=currentModel['encoderHyperparameters']
        self.decoderHyperparameters=currentModel['decoderHyperparameters']




        self.encoder=MeanScaleEncoder(self.numObsTraits+self.numLatentDx, self.nLatentDim, n_cat_list= self.numCatList, **self.encoderHyperparameters)
        self.encoder.load_state_dict(torch.load(fDirec+'/encoder.pth'))

        if self.mappingFunction=='Linear':
            self.decoder = LinearDecoder(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn)
        elif self.mappingFunction=='Linear_Monotonic':
            self.decoder=LinearDecoder_Monotonic(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn)
        elif self.mappingFunction=='Nonlinear':
            self.decoder=NonlinearMLPDecoder(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn,self.coupleCovariates,**self.decoderHyperparameters)
        elif self.mappingFunction=='Nonlinear_Monotonic':
            self.decoder=NonlinearMLPDecoder_Monotonic(self.nLatentDim,self.numCatList,self.numObsTraits,self.dropLinearCovariateColumn,**self.decoderHyperparameters)

        self.decoder.load_state_dict(torch.load(fDirec+'/decoder.pth'))
        for key,value in self.posteriorParmDict.items():
            self.posteriorParmDict[key].data = currentModel['posteriorParmDict'][key].data

        #load the pyro.param store
        pStore = pyro.get_param_store()
        pStore.load(fDirec+'/mParams.pth')



        self._setEvalMode()


if __name__ == "__main__":

    """
    code used for debugging models

    """
    from ClinicalDataset import ClinicalDataset, ClincalDatasetSampler
    from Optimizers import Optimizers
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    from sklearn import linear_model
    import shutil


    pyro.clear_param_store()
    try:
        shutil.rmtree('tmp_noDx')
        shutil.rmtree('tmp_wDx')
    except FileNotFoundError:
        pass

    numSimDiseases = 500
    numPatients = 1000
    numLatentDim = 1
    numLatentDx = 2
    numCovPerClass = [2,3,10]
    covNames = ['A','B','C']
#    numCovPerClass=[]
#    covNames=[]
#
    clinData = ClinicalDataset()
    disList = list(clinData.codeToIntMap.keys())[0:numSimDiseases+numLatentDx]
    disOfInterest = disList[-1*numLatentDx:]

    clinData.IncludeOnly(disList)
    simModel = vLPM(numSimDiseases,numCovPerClass,numLatentDx,numLatentDim,mappingFunction = 'Nonlinear',dxCodeSensitivity_95CI = [0.5,0.9],dxCodeSpecificity_95CI = [0.99,0.999],biasPrior=[-1.0,2.0])
    rawSimData = simModel.SimuateData(numPatients)

    clinData.LoadFromArrays(torch.cat((rawSimData[1],rawSimData[2]),dim=-1),rawSimData[0],covNames,catCovDicts=None, arrayType = 'Torch')
    clinData.ConditionOnDx(disOfInterest)

    pyro.clear_param_store()

#    samplerUnsup = ClincalDatasetSampler(clinData,0.75,returnArrays='Torch')
#    LPM_noDx = vLPM(numSimDiseases,numCovPerClass,0,numLatentDim,mappingFunction = 'Nonlinear')
#
#    optim = Optimizers(LPM_noDx,samplerUnsup,learningRate = 0.05,slidingErrorWindow=10)
#    outputNoDx= optim.BatchTrain(5000,2000,errorTol=1e-4)
#    X_1_unsup,X_2_unsup = samplerUnsup.ReturnFullTestingDataset(randomize=False)
#    postParams_noDx= LPM_noDx.ReturnPosteriorParameters()
#    predLatentDis_noDx,predLatentPheno_noDx = LPM_noDx.PredictTrueDisStatus(X_1_unsup,X_2_unsup)
#    trueLatentDis_unsup = rawSimData[3][samplerUnsup.testDataIndex]
#    LPM_noDx.WriteToFile('tmp_noDx')
#
#    pyro.clear_param_store()
#
#    LPM_noDx_copy = vLPM(numSimDiseases,numCovPerClass,False,numLatentDim,mappingFunction = 'Nonlinear')
#    LPM_noDx_copy.LoadParametersFromFile('tmp_noDx')
#    shutil.rmtree('tmp_noDx')
#    predLatentDis_noDx_copy,predLatentPheno_noDx_copy = LPM_noDx_copy.PredictTrueDisStatus(X_1_unsup,X_2_unsup)
#
#    optim = Optimizers(LPM_noDx_copy,samplerUnsup,learningRate = 0.05,slidingErrorWindow=10)
#    outputWDx_copy= optim.BatchTrain(5000,200,errorTol=1e-4)



    samplerSup = ClincalDatasetSampler(clinData,0.75,returnArrays='Torch',conditionSamplingOnDx=disOfInterest)
    LPM_wDx = vLPM(numSimDiseases,numCovPerClass,numLatentDx,numLatentDim,mappingFunction = 'Nonlinear',dxCodeSensitivity_95CI = [0.5,0.9])
    optim = Optimizers(LPM_wDx,samplerSup,learningRate = 0.01,slidingErrorWindow=10)
    outputWDx = optim.BatchTrain(5000,2000,errorTol=1e-4)
    postParams_wDx = LPM_wDx.ReturnPosteriorParameters()
    X_1_sup,X_2_sup,Y_sup = samplerSup.ReturnFullTestingDataset(randomize=False)
    predLatentDis_wDx,predLatentPheno_wDx = LPM_wDx.PredictTrueDisStatus(X_1_sup,X_2_sup,Y_sup)
    trueLatentDis_sup = rawSimData[3][np.concatenate(samplerSup.testDataIndex)]
    LPM_wDx.WriteToFile('tmp_wDx')
    pyro.clear_param_store()

    LPM_wDx_copy = vLPM(numSimDiseases,numCovPerClass,True,numLatentDim,mappingFunction = 'Nonlinear')
    LPM_wDx_copy.LoadParametersFromFile('tmp')
    shutil.rmtree('tmp')
    predLatentDis_wDx_copy,predLatentPheno_wDx_copy = LPM_wDx_copy.PredictTrueDisStatus(X_1_sup,X_2_sup,Y_sup)

    optim = Optimizers(LPM_wDx_copy,samplerSup,learningRate = 0.01,slidingErrorWindow=10)
    outputWDx_copy= optim.BatchTrain(5000,200,errorTol=1e-4)
#
#    logisticRegressionModel=linear_model.LogisticRegression(penalty='l2', tol=1e-6, C=1.0, fit_intercept=True, intercept_scaling=1, solver='lbfgs', max_iter=1000, verbose=0, warm_start=False)
#    X_log,Y_log = samplerSup.ReturnFullTrainingDataset_Sparse(randomize=True)
#    Y_log=np.array(Y_log.sum(dim=1).detach().numpy(),dtype=np.bool)
#    fittedModel = logisticRegressionModel.fit(X_log,Y_log)
#    X_logtest,Y_logtest = samplerSup.ReturnFullTestingDataset_Sparse(randomize=False)
#    pred_Y_log= fittedModel.predict_proba(X_logtest)[:,1:]
#
#    f, axes = plt.subplots(2, 2,figsize=(8,8))
#    for i in range(numLatentDim):
#        axes[0][1].plot(rawSimData[-2].detach().numpy().T[:,samplerUnsup.testDataIndex][i],predLatentPheno_noDx.detach().numpy().T[i],'o',color='r',alpha=0.4)
#        axes[0][1].plot(rawSimData[-2].detach().numpy().T[:,np.concatenate(samplerSup.testDataIndex,axis=-1)][i],predLatentPheno_wDx.detach().numpy().T[i],'o',color='b',alpha=0.4)
#    axes[0][1].plot(np.linspace(torch.min(rawSimData[-2]),torch.max(rawSimData[-2]),100),np.linspace(torch.min(rawSimData[-2]),torch.max(rawSimData[-2]),100),'--')
#####
#    axes[0][0].plot(outputNoDx[0],color='r')
#    axes[0][0].plot(outputWDx[0],color='b')
##
#    axes[1][0].plot(rawSimData[-1]['bias'].detach().numpy(),postParams_noDx['biasPosteriorMean'].detach().numpy(),'o',color='r',alpha=0.4)
#    axes[1][0].plot(rawSimData[-1]['bias'].detach().numpy(),postParams_wDx['biasPosteriorMean'].detach().numpy(),'o',color='b',alpha=0.4)
#    axes[1][0].plot(np.linspace(torch.min(rawSimData[-1]['bias']),torch.max(rawSimData[-1]['bias']),100),np.linspace(torch.min(rawSimData[-1]['bias']),torch.max(rawSimData[-1]['bias']),100),'k--')
#
#
#    precisionRecall = precision_recall_curve(trueLatentDis_sup.detach().numpy(),pred_Y_log)
#    axes[1][1].step(precisionRecall[1], precisionRecall[0], color='g', alpha=0.5,where='post',lw=3.0)
###
####
#    precisionRecall = precision_recall_curve(trueLatentDis_sup.detach().numpy(),predLatentDis_wDx.detach().numpy())
#    axes[1][1].step(precisionRecall[1], precisionRecall[0], color='b', alpha=0.5,where='post',lw=3.0)
####
#    precisionRecall = precision_recall_curve(trueLatentDis_unsup.detach().numpy(),predLatentDis_noDx.detach().numpy())
#    axes[1][1].step(precisionRecall[1], precisionRecall[0], color='r', alpha=0.5,where='post',lw=3.0)
#
#    plt.show()
#    for i in range(numLatentDx):
#        print('Dx Code Sensitivity (Sim, Inf): {}, {}'.format(rawSimData[-1]['sensitivity'][:,i],postParams_wDx['dxCodeSensitivityPosterior'][0,i]/postParams_wDx['dxCodeSensitivityPosterior'][:,i].sum()))
#        print('Dx Code Specificity (Sim, Inf): {}, {}'.format(rawSimData[-1]['specificity'][:,i],postParams_wDx['dxCodeSpecificityPosterior'][0,i]/postParams_wDx['dxCodeSpecificityPosterior'][:,i].sum()))
