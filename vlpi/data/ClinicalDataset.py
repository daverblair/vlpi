#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:37:15 2019

@author: davidblair
"""

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from scipy import sparse
from typing import Iterable
from collections import OrderedDict
from sklearn.utils import shuffle
import copy
import pickle
import itertools
from vlpi.utils.UtilityFunctions import one_hot_scipy,one_hot
from vlpi.data.ICDUtilities import ICDUtilities

class ClinicalDataset:

    def none_to_int(self,val):
        if val!=None:
            return 1
        else:
            return 0

    def initialize_empty_df(self,columns, dtypes, index=None):
        assert len(columns)==len(dtypes)
        df = pd.DataFrame(index=index)
        for c,d in zip(columns, dtypes):
            df[c] = pd.Series(dtype=d)
        return df

    def _parseICDCodeList(self,codeList):
        codeList = list(set(codeList.strip().split(',')))
        codeList=[x.strip() for x in codeList if x.strip()!='']

        newCodeList=[]
        for code in codeList:
            try:
                newCodeList+=[self.dxCodeToDataIndexMap[code]]
            except KeyError:
                pass

        return newCodeList

    def _parseCatCov(self,covList,catCovNames):
        covVals = []
        for i,covName in enumerate(catCovNames):
            try:
                covVals+=[(covName,self.catCovConversionDicts[covName][covList[i]])]
            except KeyError:
                newIndex = len(self.catCovConversionDicts[covName])
                self.catCovConversionDicts[covName][covList[i]]=newIndex
                covVals+=[(covName,newIndex)]
        return covVals


    def __init__(self,ICDFilePaths:Iterable[str]=[]):
        """


        Parameters
        ----------
        ICDFilePaths : Iterable[str], optional
            This passes a list of strings ([ICD_hierarchy, ICD_chapters], see ICDUtilities) in order to initialize Dx Code data structure and mappings. This is only relevant when constructing new datasets from a flat text file. Otherwise, the Dx Code information is read from the stored ClinicalDataset Object, so the file paths are irrelevant. By default, the class instantiates the 2018 ICD10-CM coding structure, which is included with the software (as is the UKBB ICD10 encoding structure, downloaded in Jan 2020).

            The default is value [], which defaults to ICD10-CM 2018.

        Returns
        -------
        None.

        """



        if len(ICDFilePaths)==0:
            self.ICDInfo=ICDUtilities()
        else:
            assert len(ICDFilePaths)==2, "Expects a list containing 2 elecments: file paths for ICD10 hierarchy and chapters"
            self.ICDInfo=ICDUtilities(hierarchyFile=ICDFilePaths[0],chapterFile=ICDFilePaths[1])

        #initialize the clinical data structure to line up with the ICD codebook,
        #although the clinical dataset need not correspond strictly to ICD codes (and this is frequently true)
        self.dxCodeToDataIndexMap = copy.deepcopy(self.ICDInfo.usableCodeToIndexMap)
        self.dataIndexToDxCodeMap = dict(zip(self.dxCodeToDataIndexMap.values(),self.dxCodeToDataIndexMap.keys()))

        self.numDxCodes = len(self.dxCodeToDataIndexMap)
        self.data=None
        self.numPatients = None
        self.catCovConversionDicts={}

    def ReadDatasetFromFile(self,clinicalDataset,dxCodeColumn,indexColumn = None, skipColumns=[], hasHeader=True,chunkSize = 500):
        """
        Initializes the Pandas clinical dataset by reading it from a text file.

        Expects that clinical dataset is in ICD format. Can transition to other formats (HPO)
        by using using ConvertCodes function.

        clinicalDataset-->fileName for clinical dataset
        dxCodeColumn-->column that contains a comma-separated list of associated ICD codes, first column denoted by 0
        indexColumn-->column to use as index for the dataset
        skipColumns-->list that indicates which columns should be skipped [uses 0-based indexing]
        has_header-->indicates whether file has header, which is used to generate column names
        chunkSize-->indicates how often database should be written into. Defaults to every 500 lines.
        """

        assert chunkSize >1, "chunkSize must be > 1"
        clinicalFile = open(clinicalDataset)
        if hasHeader:
            headLine = clinicalFile.readline().strip('\n').split('\t')
            catCovNames = [h for h in headLine if headLine.index(h) not in [dxCodeColumn,indexColumn]+skipColumns]
        else:
            currentLine = clinicalFile.readline().strip('\n').split('\t')
            catCovNames=['Covariate_'+str(i+1) for i in range(len(currentLine)-(1+self.none_to_int(indexColumn)+len(skipColumns)))]

        colNames = ['patient_id','dx_codes']+catCovNames

        self.catCovConversionDicts = {covName:{} for covName in catCovNames}
        self.data = self.initialize_empty_df(colNames,[np.int64,object]+[np.int32 for i in range(len(catCovNames))])
        patientCounter = int(0)
        currentDataList ={colName:[] for colName in self.data.columns}
        for line in clinicalFile:
            line = line.strip('\n').split('\t')
            currentDataList['dx_codes']+=[self._parseICDCodeList(line[dxCodeColumn])]
            for nm, val in self._parseCatCov([line[i] for i in range(len(line)) if i not in [dxCodeColumn,indexColumn]+skipColumns],catCovNames):
                currentDataList[nm]+=[val]

            if indexColumn!=None:
                currentDataList['patient_id']+=[int(line[indexColumn])]
            else:
                currentDataList['patient_id']+=[patientCounter]
            patientCounter+=1
            if patientCounter % chunkSize==0:
                self.data=self.data.append(pd.DataFrame(currentDataList),ignore_index=True)
                currentDataList ={colName:[] for colName in self.data.columns}


        if len(currentDataList['patient_id'])>0:
            self.data=self.data.append(pd.DataFrame(currentDataList),ignore_index=True)

        #shuffle data and create new index
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data.set_index('patient_id',drop=False, inplace=True)
        self.numPatients = len(self.data)


    def FindAllPatients_wDx(self,dx_code):
        """
        Finds all patients with a particular dx code, returns their index vals
        """

        if '.' in dx_code:
            dx_code.replace('.','')
        intVal = self.dxCodeToDataIndexMap[dx_code]
        return self.data['patient_id'][self.data['dx_codes'].apply(lambda x: intVal in x)]


    def IncludeOnly(self,dx_code_list):
        """
        Removes all dx_codes from the dataset except those from the dx_code_list
        """

        dx_code_list=[x.replace('.','') for x in dx_code_list]
        allKept = set([self.dxCodeToDataIndexMap[x] for x in dx_code_list])

        #now we need to remove all non-kept codes from the ICD conversion dictionaries
        removedCodes = set(self.dxCodeToDataIndexMap.keys()).difference(dx_code_list)

        for old_code in removedCodes:
            del self.dxCodeToDataIndexMap[old_code]

        self.dataIndexToDxCodeMap = {}
        newCodeToIntMap = {}
        oldToNewIntMap={}
        for i,key in enumerate(self.dxCodeToDataIndexMap):
            oldToNewIntMap[self.dxCodeToDataIndexMap[key]]=i
            self.dataIndexToDxCodeMap[i]=key
            newCodeToIntMap[key] = i

        self.dxCodeToDataIndexMap=newCodeToIntMap
        if isinstance(self.data,pd.DataFrame):
            self.data['dx_codes']=self.data['dx_codes'].apply(lambda x: [oldToNewIntMap[y] for y in x if y in allKept])

        self.numDxCodes=len(self.dxCodeToDataIndexMap)

    def ConvertCodes(self,dx_code_list:Iterable[str],new_code:str):
        """
        Converts set of ICD codes into a single dx code through logical-OR function.
        If given a single code, simply renames code as new_code.

        dx_code_list-->List of codes to collapse
        new_string-->new string to name the dx
        new_code-->new code to correspond to the string. Will automatically be converted into an integer value
        """
        assert len(dx_code_list)>0, "dx_code_list must have one elemenent to collapse."
        dx_code_list=[x.replace('.','') for x in dx_code_list]

        #set the all codes in the list to the integer value of the first code in the list
        removedCodes = set(dx_code_list[1:])
        oldInts =[]
        for old_code in removedCodes:
            oldInts+=[self.dxCodeToDataIndexMap[old_code]]
            del self.dxCodeToDataIndexMap[old_code]


        if len(removedCodes)>0:
            self.dataIndexToDxCodeMap = {}
            newCodeToIntMap = {}
            oldToNewIntMap={}
            for i,key in enumerate(self.dxCodeToDataIndexMap):
                oldToNewIntMap[self.dxCodeToDataIndexMap[key]]=i
                self.dataIndexToDxCodeMap[i]=key
                newCodeToIntMap[key] = i

            self.dxCodeToDataIndexMap=newCodeToIntMap
            newInt = self.dxCodeToDataIndexMap[dx_code_list[0]]
            collapsedInt_to_Int = dict(zip(oldInts,[newInt for i in range(len(oldInts))]))
            oldToNewIntMap.update(collapsedInt_to_Int)
            if isinstance(self.data,pd.DataFrame):
                self.data['dx_codes']=self.data['dx_codes'].apply(lambda x: list(set([oldToNewIntMap[y] for y in x])))
        else:
            newInt = self.dxCodeToDataIndexMap[dx_code_list[0]]


        #update the code information

        self.dataIndexToDxCodeMap[newInt] = new_code
        self.dxCodeToDataIndexMap[new_code] = newInt
        del self.dxCodeToDataIndexMap[dx_code_list[0]]
        self.numDxCodes=len(self.dxCodeToDataIndexMap)


    def ConstructNewDataArray(self,oldCodeToNewMap):
        """
        This function translates diagnostic codes using dictionary oldCodeToNewMap.
        Codes not in this dictionary are dropped.
        It reqirues the following argurments:
            oldCodeToNewMap-> key,value pairs indicating translation of old codes to new, can be one to many but expects iterable
        """
        allNewCodes = sorted(list(set().union(*oldCodeToNewMap.values())))
        newCodeToIntMap = dict(zip(allNewCodes,range(len(allNewCodes))))
        newIntToCodeMap = dict(zip(newCodeToIntMap.values(),newCodeToIntMap.keys()))

        def _convFunc(x):
            newDxSet=set([])
            for dx in x:
                try:
                    newDxSet.update(oldCodeToNewMap[self.dataIndexToDxCodeMap[dx]])
                except KeyError:
                    pass
            return list({newCodeToIntMap[x] for x in newDxSet})

        self.data['dx_codes'] = self.data['dx_codes'].apply(_convFunc)
        self.dataIndexToDxCodeMap = newIntToCodeMap
        self.dxCodeToDataIndexMap = newCodeToIntMap
        self.numDxCodes=len(self.dxCodeToDataIndexMap)

    def ExcludeAll(self,dx_code_list):
        """
        Removes all codes in dx_code_list from the dataset

        """
        keptCodes = set(self.dxCodeToDataIndexMap.keys()).difference(dx_code_list)
        self.IncludeOnly(list(keptCodes))

    def ConditionOnDx(self,dx_code_list):
        """
        Conditions the data table on whether a patient has a particular or set of dx's
        Accomplished by finding all patients with a each dx,
        then adding a column (boolean) to the data table indicating this to be true.
        Completed by removing the conditioned codes from the data table.
        This function is expecially useful for supervised learning, as it allows
        easy conditioned sampling from the data.
        """
        for dx_code in dx_code_list:
            dx_code.replace('.','')
            allPatients_wDx=self.FindAllPatients_wDx(dx_code)
            hasDx=np.zeros(self.numPatients,dtype=np.bool)
            self.data.insert(len(self.data.columns),'has_'+dx_code,hasDx)
            self.data.loc[allPatients_wDx,'has_'+dx_code]=True
        self.ExcludeAll(dx_code_list)



    def WriteToDisk(self,fileName):
        """
        Writes to disk using pickle.
        """
        if fileName[-4:]!='.pth':
            fileName+='.pth'
        with open(fileName,'wb') as f:
            pickle.dump(self.data,f,protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.catCovConversionDicts,f,protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dxCodeToDataIndexMap,f,protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataIndexToDxCodeMap,f,protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.numDxCodes,f,protocol=pickle.HIGHEST_PROTOCOL)

    def ReadFromDisk(self,fileName):
        """
        Reads from disk using pickle
        """
        if fileName[-4:]!='.pth':
            fileName+='.pth'
        with open(fileName,'rb') as f:
            self.data = pickle.load(f)
            self.catCovConversionDicts = pickle.load(f)
            self.dxCodeToDataIndexMap = pickle.load(f)
            self.dataIndexToDxCodeMap = pickle.load(f)
            self.numDxCodes = pickle.load(f)

        self.numPatients = len(self.data)


    def ReadFromHDF5(self,fileName):

        """
        Reads dataset from hdf5 format. No storage of meta-data. Need to make sure ClinicalDatasets
        have identical ICD10 codes and funtions. Use with caution. Also, not thread safe.
        """
        if fileName[-3:]!='.h5':
            fileName+='.h5'

        with pd.HDFStore(fileName) as dataFile:
            self.data = dataFile['data']
            catCovNames = list(self.data.columns)
            catCovNames.remove('patient_id')
            catCovNames.remove('dx_codes')
            for name in catCovNames:
                tmpTable = dataFile['/catCovConversionDicts/'+name]
                self.catCovConversionDicts[name]=dict(zip(tmpTable['OrigVal'],tmpTable['IntVal']))
            self.dxCodeToDataIndexMap = dataFile['dxCodeToDataIndexMap']
            self.dataIndexToDxCodeMap = dataFile['dataIndexToDxCodeMap']

        self.numPatients = len(self.data)
        self.numDxCodes = len(self.dxCodeToDataIndexMap)


    def LoadFromArrays(self,incidenceArray,covariateArrays,covariateNames,catCovDicts=None, arrayType = 'Numpy'):
        """
        Loads clinical dataset from array data, used for simulation purposes.

        incidenceArray-->numSamples x numDx array of binary values indicating dx presence/abscence
        covariateArrays-->List of covariate vectors, each numSamples in length. Values can be arbitrary
        covariateNames-->names for the covariates contained within covariateArrays
        catCovDicts-->optional: supply a list of dictionaries (one for each covariate) that maps covariates to integer values
        arrayType-->Numpy arrays ['Numpy'] or pyTorch tensors ['Torch']
        """
        assert arrayType in ['Numpy','Torch'], "Only Numpy arrarys or Torch tensors supported"
        if covariateArrays==None:
            covariateArrays=[]
        if covariateNames==None:
            covariateNames=[]
        assert len(covariateArrays)==len(covariateNames), "Number of covariate names does not match number of covariate arrays."
        assert incidenceArray.shape[1]==self.numDxCodes, "Dimension of incidence data does not match number of codes."

        if arrayType=='Torch':
            incidenceArray=incidenceArray.to('cpu').detach().numpy()
            covariateArrays=[x.to('cpu').detach().numpy().ravel() for x in covariateArrays]
        else:
            covariateArrays=[x.ravel() for x in covariateArrays]

        dataDict={}
        for i,name in enumerate(covariateNames):
            if catCovDicts == None:
                uniqueCats = list(set(covariateArrays[i]))
                self.catCovConversionDicts[name] = dict(zip(uniqueCats,list(range(len(uniqueCats)))))
                covariateArrays[i] = np.array([self.catCovConversionDicts[name][x] for x in covariateArrays[i]],dtype=np.int64)
            else:
                self.catCovConversionDicts[name]=catCovDicts[i]
                covariateArrays[i] = np.array([self.catCovConversionDicts[name][x] for x in covariateArrays[i]],dtype=np.int64)
            dataDict[name] = covariateArrays[i]

        dataDict['patient_id']=np.arange(incidenceArray.shape[0],dtype=np.int64)
        dataDict['dx_codes'] = [np.where(x==1)[0].tolist() for x in incidenceArray]

        self.data = pd.DataFrame(dataDict)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data.set_index('patient_id',drop=False, inplace=True)
        self.numPatients = len(self.data)

    def ReturnSparseDataMatrix(self,index:Iterable[int]=[]):
        """
        Returns disease incidence array as sparse coo matrix. Takes optional index, which returns
        only data points contained within the index.
        """
        if len(index)==0:
            index = self.data.index
        y_inds = list(itertools.chain.from_iterable(self.data.loc[index]['dx_codes']))
        x_inds = list(itertools.chain.from_iterable([[i]*len(x) for i,x in enumerate(self.data.loc[index]['dx_codes'])]))
        return sparse.coo_matrix((np.ones((len(x_inds))),(x_inds,y_inds)),shape=(len(index),self.numDxCodes),dtype=np.float32)




class ClinicalDatasetSampler():
    """
    Generates random samples from a clinical dataset. Samples can be generated
    unconditionially, or conditional on a patient having a particular dx.
    Note, that in the latter case, the dx will be removed from the dataset and included as
    a separate column in the data if not already done.

    """

    def _numpyWrapper(self,x):
        if x.dtype == np.float32:
            if sparse.issparse(x):
                return x.toarray()
            else:
                return np.array(x,dtype=np.float32)

        elif x.dtype == np.float64:
            if sparse.issparse(x):
                return x.toarray()
            else:
                return np.array(x,dtype=np.float64)
        else:
            if sparse.issparse(x):
                return x.toarray()
            else:
                return np.array(x,dtype=np.int64)

    def _scipySparseWrapper(self,x):
        if x.dtype == np.float32:
            if sparse.issparse(x):
                return x.tocsr()
            else:
                return sparse.csr_matrix(x,dtype=np.float32)

        elif x.dtype==np.float64:
            if sparse.issparse(x):
                return x.tocsr()
            else:
                return sparse.csr_matrix(x,dtype=np.float64)
        else:
            if sparse.issparse(x):
                return x.tocsr()
            else:
                return sparse.csr_matrix(x,dtype=np.int64)

    def _torchWrapper(self,x):

        """
        Note, all torch floating point tensors are converted to 32-bits to
        ensure GPU compatibility.
        """

        if x.dtype==np.float32:
            if sparse.issparse(x):
                return torch.tensor(x.toarray(),dtype = torch.float32)
            else:
                return torch.tensor(x,dtype = torch.float32)

        elif x.dtype==np.float64:
            if sparse.issparse(x):
                return torch.tensor(x.toarray(),dtype = torch.float32)
            else:
                return torch.tensor(x,dtype = torch.float32)
        else:
            if sparse.issparse(x):
                return torch.tensor(x.toarray(),dtype = torch.long)
            else:
                return torch.tensor(x,dtype = torch.long)



    def __init__(self, currentClinicalDataset,trainingFraction,conditionSamplingOnDx:Iterable[str]=[],returnArrays='Numpy',shuffle=True):
        """
        Initializes the clinical dataset sampler class, which uses a sparse representation of the
        underlying data matrix to draw samples quickly.

        currentClinicalDataset-->instance of the class Clinical Dataset
        trainingFraction-->fraction of dataset used for training
        conditionSamplingOnDx-->allows sampling to be conditioned on a set of dx
            such that at least one patient in every sample had at least one of the dx in the set.
            Note: original dataset is modified.
        returnArrays-->array type returned by the sampling. Can be 'Numpy', 'Sparse' or 'Torch'
            In the case of Sparse arrays, incidence arrays are returned as csr matrices, 1-d covariate vectors default to COO format.
        shuffle-->Indcates whether to shuffle the data prior to splitting into training and test sets
            Defaults to True, only make False for very large datasets that have already been shuffled.

        """
        self.conditionSamplingOnDx=conditionSamplingOnDx
        if len(conditionSamplingOnDx)>0:
            self.isConditioned = True
        else:
            self.isConditioned = False
        self.currentClinicalDataset=currentClinicalDataset
        self._returnAuxData=False
        self._auxDataset=None
        self.trainingFraction = trainingFraction
        self.fracWDx=0.0
        self.numTotalSamples = len(self.currentClinicalDataset.data)
        self.includedCovariates = self.currentClinicalDataset.catCovConversionDicts.keys()


        assert returnArrays in ['Numpy','Torch','Sparse'], "Only Numpy arrarys, Torch tensors, or Scipy.Sparse (csr) supported"

        self.returnArrays=returnArrays
        if returnArrays =='Numpy':
            self.arrayFunc = self._numpyWrapper
        elif returnArrays =='Torch':
            self.arrayFunc=self._torchWrapper
        else:
            self.arrayFunc=self._scipySparseWrapper




        if shuffle==True:
            self.currentClinicalDataset.data=self.currentClinicalDataset.data.sample(frac=1)

        if len(self.conditionSamplingOnDx)==0:
            self.currentClinicalDataset = currentClinicalDataset
            cutOffVal = int(np.floor(len(currentClinicalDataset.data)*self.trainingFraction))
            self.trainingDataIndex = currentClinicalDataset.data.index[0:cutOffVal]
            self.testDataIndex = currentClinicalDataset.data.index[cutOffVal:]
        else:
            conditionedColumns = set(['has_'+dx_code for dx_code in self.conditionSamplingOnDx])
            missingColumns  = conditionedColumns.difference(self.currentClinicalDataset.data.columns)

            if len(missingColumns)>0:
                self.currentClinicalDataset.ConditionOnDx([x.replace('has_','') for x in missingColumns])

            has_at_least_one_dx = np.array(np.sum(np.vstack([self.currentClinicalDataset.data['has_'+dx] for dx in self.conditionSamplingOnDx]),axis=0),dtype=np.bool)
            dataWithDx = self.currentClinicalDataset.data.index[has_at_least_one_dx>0]
            dataWithoutDx = self.currentClinicalDataset.data.index[has_at_least_one_dx==0]
            self.fracWDx = len(dataWithDx)/len(self.currentClinicalDataset.data)
            cutOffValWDx = int(np.floor(len(dataWithDx)*self.trainingFraction))
            cutOffValWoDx = int(np.floor(len(dataWithoutDx)*self.trainingFraction))

            self.trainingDataIndex=[dataWithDx[0:cutOffValWDx],dataWithoutDx[0:cutOffValWoDx]]
            self.testDataIndex=[dataWithDx[cutOffValWDx:],dataWithoutDx[cutOffValWoDx:]]

    def ChangeArrayType(self,newArrayType):
        assert newArrayType in ['Numpy','Torch','Sparse'], "Only Numpy arrarys, Torch tensors, or Scipy.Sparse (csr) supported"
        self.returnArrays=newArrayType
        if newArrayType =='Numpy':
            self.arrayFunc = self._numpyWrapper
        elif newArrayType =='Torch':
            self.arrayFunc=self._torchWrapper
        else:
            self.arrayFunc=self._scipySparseWrapper



    def WriteToDisk(self,fName):
        """
        Writes sampler to disk so that it can be re-instantiated for further use.
        """
        if fName[-4:]!='.pth':
            fName+='.pth'
        currentSampler = OrderedDict()
        currentSampler['conditionSamplingOnDx']=self.conditionSamplingOnDx
        currentSampler['numTotalSamples'] = self.numTotalSamples
        currentSampler['trainingDataIndex']=self.trainingDataIndex
        currentSampler['testDataIndex']=self.testDataIndex
        currentSampler['trainingFraction']=self.trainingFraction
        currentSampler['fracWDx']=self.fracWDx


        with open(fName, 'wb') as f:
            pickle.dump(currentSampler,f)



    def ReadFromDisk(self,fName):
        """
        Reads sampler from disk.
        """
        if fName[-4:]!='.pth':
            fName+='.pth'

        with open(fName, 'rb') as f:
            currentSampler = pickle.load(f)
        #assertions to make sure that sampler parameters  match up

        assert currentSampler['numTotalSamples'] == self.numTotalSamples, "Datasets used by saved and current sampler are different lengths! Suspect they are not referring to same dataset"
        assert currentSampler['conditionSamplingOnDx']==self.conditionSamplingOnDx, "Saved and current sampler are not conditioned on same dx"
        assert currentSampler['trainingFraction']==self.trainingFraction,"Saved and current sampler do not have the same training fraction"
        self.testDataIndex=currentSampler['testDataIndex']
        self.trainingDataIndex=currentSampler['trainingDataIndex']
        self.fracWDx=currentSampler['fracWDx']

    def ConvertToUnconditional(self):
        """
        Converts a previously conditional sampler to unconditional while keeping the same testing and training sets.
        Note, if unconditional, disease that is conditioned on won't be part of the data array.
        """
        assert len(self.conditionSamplingOnDx)!=0, "Sampler is already uncoditional and was never conditional to start."
        assert isinstance(self.trainingDataIndex,list) is True, "Sampler has already been converted to unconditional."
        self.trainingDataIndex=np.concatenate(self.trainingDataIndex)
        self.testDataIndex=np.concatenate(self.testDataIndex)
        self.isConditioned=False

    def RevertToConditional(self):
        """
        Reverts a previously conditional sampler back to conditinal sampling after becoming unconditional.
        Note, a sampler originally contstructed as unconditional cannot be made conditional,
        as the make up of the testing and training datasets will not be correct.
        """
        assert len(self.conditionSamplingOnDx)!=0, "Sampler was not constructed as a conditional sampler. If you want a conditional sampler for this dataset, create a new ClincalDatasetSampler instance."
        assert isinstance(self.trainingDataIndex,list) is False, "Sampler is already conditional."
        has_at_least_one_dx_train= np.array(np.sum(np.vstack([self.currentClinicalDataset.data.loc[self.trainingDataIndex]['has_'+dx] for dx in self.conditionSamplingOnDx]),axis=0),dtype=np.bool)
        has_at_least_one_dx_test= np.array(np.sum(np.vstack([np.array(self.currentClinicalDataset.data.loc[self.testDataIndex]['has_'+dx],dtype=np.bool) for dx in self.conditionSamplingOnDx]),axis=0),dtype=np.bool)
        self.trainingDataIndex=[self.trainingDataIndex[has_at_least_one_dx_train],self.trainingDataIndex[np.invert(has_at_least_one_dx_train)]]
        self.testDataIndex=[self.testDataIndex[has_at_least_one_dx_test],self.testDataIndex[np.invert(has_at_least_one_dx_test)]]
        self.isConditioned=True

    def SubsetCovariates(self,newCovList):
        """


        Parameters
        ----------
        newCovList : List
            List of covariates contained within ClinicalDataset that should be returned by the sampler. Can be empty list, which indicates that no covariates should be returned.

        Returns
        -------
        None.

        """
        assert set(newCovList).issubset(self.includedCovariates), "Subset of covariates provided is not subset  of current covariates."
        self.includedCovariates=newCovList


    def _returnData(self,newIndex,collapseAnchorDx=True):
        if isinstance(newIndex,Iterable)==False:
            newIndex=[newIndex]

        incidenceData = self.arrayFunc(self.currentClinicalDataset.ReturnSparseDataMatrix(newIndex))
        covData = [self.currentClinicalDataset.data.loc[newIndex][s].values for s in self.includedCovariates]
        covData = [self.arrayFunc(x.reshape(incidenceData.shape[0],1)) for x in covData]

        if not self.isConditioned:
            target_data = None
        else:
            target_data = np.array(pd.concat([self.currentClinicalDataset.data.loc[newIndex]['has_'+dx] for dx in self.conditionSamplingOnDx],axis=1),dtype=np.float32)
            target_data=self.arrayFunc(target_data.reshape(incidenceData.shape[0],len(self.conditionSamplingOnDx)))


        if not self._returnAuxData:
            encoded_data = None
        else:
            encoded_data = self.arrayFunc(self._auxDataset.ReturnSparseDataMatrix(newIndex))

        return incidenceData,covData,target_data,encoded_data

    def _generateRandomSample(self,numSamples,datasetIndex,fixedFracWDx):
        if fixedFracWDx!=None:
            assert isinstance(fixedFracWDx, float) and fixedFracWDx < 1.0 and fixedFracWDx > 0.0, "fixedFrac with dx must be float between 0.0 and 1.0"
            assert len(self.conditionSamplingOnDx)>0, "Cannot include fixed fraction of positive cases if conditionSamplingOnDx not enabled"
            if fixedFracWDx*numSamples >= len(datasetIndex[0]):
                numSamplesWDx = len(datasetIndex[0])
                print("Warning: fixedFracWDx exceeds or is equal to the total number of positive training samples.\nEvery positive case will be included in random sample")
            else:
                numSamplesWDx = int(np.ceil(numSamples*fixedFracWDx))
        elif self.isConditioned:
            numSamplesWDx=int(np.ceil(numSamples*self.fracWDx))


        if not self.isConditioned:
            newIndex = np.random.choice(datasetIndex,size=numSamples,replace=False)
        else:
            newIndex = shuffle(np.concatenate((np.random.choice(datasetIndex[0],size=numSamplesWDx,replace=False),np.random.choice(datasetIndex[1],size=(numSamples-numSamplesWDx),replace=False))))
        return self._returnData(newIndex)


    def GenerateRandomTrainingSample(self,numSamples, fixedFracWDx=None):
        return self._generateRandomSample(numSamples,self.trainingDataIndex,fixedFracWDx)


    def GenerateRandomTestSample(self,numSamples,fixedFracWDx=None):
        return self._generateRandomSample(numSamples,self.testDataIndex,fixedFracWDx)

    def _returnFullDataset(self,datasetIndex,randomize):
        if self.isConditioned:
            datasetIndex = np.concatenate(datasetIndex,axis=0)
        if randomize==True:
            datasetIndex=shuffle(datasetIndex)

        return self._returnData(datasetIndex)


    def ReturnFullTrainingDataset(self,randomize=True):
        return self._returnFullDataset(self.trainingDataIndex,randomize)

    def ReturnFullTestingDataset(self,randomize=True):
        return self._returnFullDataset(self.testDataIndex,randomize)


    def _indexSplit(self,dataSetSize,totalNumBatches):
        if totalNumBatches>0:
            nEachBatch, extras = divmod(dataSetSize, totalNumBatches)
            section_sizes = ([0] +extras * [nEachBatch+1] +(totalNumBatches-extras) * [nEachBatch])
            return np.array(section_sizes, dtype=np.int32).cumsum()
        else:
            return np.array([0]+[dataSetSize], dtype=np.int32)


    def _epoch(self,datasetIndex,batch_size):

        if self.isConditioned:
            #shuffle the order of the dataset at the start of every epoch
            datasetIndex[0]=np.random.permutation(datasetIndex[0])
            datasetIndex[1]=np.random.permutation(datasetIndex[1])
            totalNumBatches,leftover = divmod((len(datasetIndex[0])+len(datasetIndex[1])),batch_size)
            assert totalNumBatches <= len(datasetIndex[0]), "Batch size too small. Cannot ensure at least one positive example per batch."
            #first shuffle the data

            hasDxSplits=self._indexSplit(len(datasetIndex[0]),totalNumBatches)
            noDxSplits=self._indexSplit(len(datasetIndex[1]),totalNumBatches)
            if totalNumBatches == 0:
                totalNumBatches+=1

            for i in range(totalNumBatches):
                hasDxSubset = datasetIndex[0][hasDxSplits[i]:hasDxSplits[i+1]]
                noDxSubset = datasetIndex[1][noDxSplits[i]:noDxSplits[i+1]]

                batchIndex = np.concatenate((hasDxSubset,noDxSubset))
                #need to shuffle positive cases within all others
                yield shuffle(batchIndex)
        else:
            datasetIndex=shuffle(datasetIndex)
            totalNumBatches,leftover = divmod(len(datasetIndex),batch_size)
            splits = self._indexSplit(len(datasetIndex),totalNumBatches)
            if totalNumBatches == 0:
                totalNumBatches+=1
            for i in range(totalNumBatches):

                yield datasetIndex[splits[i]:splits[i+1]]


    def TrainingEpochGenerator(self, batch_size):
        for batch in self._epoch(self.trainingDataIndex,batch_size):
            yield self._returnData(batch)


    def TestingEpochGenerator(self,batch_size):
        for batch in self._epoch(self.testDataIndex,batch_size):
            yield self._returnData(batch)

    def GenerateValidationSampler(self,validation_fraction):
        """
        Returns a new ClinicalDatasetSampler that splits the training set
        into training and validation sets. This new sampler can be used just like the original
        except that the test data is now the validation data. It accomplishes this task
        by making a general shallow copy of the class (to avoid copying, for example, the whole dataset)
        while making deep copies of the information that changes between the
        validation and test classes.
        """
        new_instance = copy.copy(self)
        new_instance.trainingDataIndex=copy.deepcopy(self.trainingDataIndex)
        new_instance.testDataIndex=copy.deepcopy(self.testDataIndex)
        new_instance.trainingFraction=copy.deepcopy(self.trainingFraction)
        new_instance.trainingFraction = 1.0-validation_fraction

        if self.isConditioned==False:
            new_instance.numTotalSamples=len(self.trainingDataIndex)
            trainingDataIndexShuffled = shuffle(self.trainingDataIndex)
            cutOffVal = int(np.floor(len(trainingDataIndexShuffled)*new_instance.trainingFraction))
            new_instance.trainingDataIndex = trainingDataIndexShuffled[0:cutOffVal]
            new_instance.testDataIndex = trainingDataIndexShuffled[cutOffVal:]
        else:
            new_instance.numTotalSamples=len(self.trainingDataIndex[0])+len(self.trainingDataIndex[1])
            trainingDataIndexShuffled = np.append(*[shuffle(np.array(x)) for x in self.trainingDataIndex])

            has_at_least_one_dx = np.array(np.sum(np.vstack([self.currentClinicalDataset.data.loc[trainingDataIndexShuffled]['has_'+dx] for dx in self.conditionSamplingOnDx]),axis=0),dtype=np.bool)
            dataWithDx = self.currentClinicalDataset.data.loc[trainingDataIndexShuffled].index[has_at_least_one_dx>0]
            dataWithoutDx = self.currentClinicalDataset.data.loc[trainingDataIndexShuffled].index[has_at_least_one_dx==0]
            self.fracWDx = len(dataWithDx)/len(trainingDataIndexShuffled)
            cutOffValWDx = int(np.floor(len(dataWithDx)*new_instance.trainingFraction))
            cutOffValWoDx = int(np.floor(len(dataWithoutDx)*new_instance.trainingFraction))

            new_instance.trainingDataIndex=[dataWithDx[0:cutOffValWDx],dataWithoutDx[0:cutOffValWoDx]]
            new_instance.testDataIndex=[dataWithDx[cutOffValWDx:],dataWithoutDx[cutOffValWoDx:]]
        return new_instance

    def CollapseDataArrays(self,disInds=None,cov_vecs=None,drop_column=False):
        list_of_arrays=[]
        if disInds is not None:
            list_of_arrays+=[disInds]


        if cov_vecs is not None:
            n_cat_vec = [len(self.currentClinicalDataset.catCovConversionDicts[x]) for x in self.includedCovariates]

            for i,n_cat in enumerate(n_cat_vec):
                if torch.is_tensor(cov_vecs[0]):
                    list_of_arrays+=[one_hot(cov_vecs[i],n_cat,dropColumn=drop_column)]
                else:
                    list_of_arrays+=[one_hot_scipy(cov_vecs[i],n_cat,dropColumn=drop_column)]


        list_of_arrays=[self.arrayFunc(x) for x in list_of_arrays]

        if self.returnArrays=='Numpy':
            return np.hstack(list_of_arrays,dtype=np.float64)
        elif self.returnArrays=='Sparse':
            return sparse.hstack(list_of_arrays,format='csr',dtype=np.float64)
        else:
            return torch.cat(list_of_arrays,dim=1,dtype=torch.float32)

    def AddAuxillaryDataset(self,newClinicalDataset):
        assert len(self.currentClinicalDataset.data.index.difference(newClinicalDataset.data.index))==0,"Auxillary ClinicalDataset must contain the same samples as the original ClinicalDataset"
        self._returnAuxData=True
        self._auxDataset=newClinicalDataset



    def RemoveAuxillaryDataset(self):
        self._returnAuxData=False
        self._auxDataset=None

class _TorchDatasetWrapper(data.Dataset):

    def __init__(self,clinicalDatasetSampler,sampling_index,batch_size):
        """
        Wrapper for ClinicalData and ClinicalDatasetSampler to allow for
        rapid subset sampling using DataLoader, which allows for multi-thread
        loading/queueing of data.
        """

        self.clinicalDatasetSampler = clinicalDatasetSampler
        self.sampling_index=sampling_index


        if self.clinicalDatasetSampler.isConditioned:
            #shuffle the order of the dataset at the start of every epoch
            self.sampling_index[0]=shuffle(self.sampling_index[0])
            self.sampling_index[1]=shuffle(self.sampling_index[1])
            self.totalNumBatches,leftover = divmod((len(self.sampling_index[0])+len(self.sampling_index[1])),batch_size)
            assert self.totalNumBatches <= len(sampling_index[0]), "Batch size too small. Cannot ensure at least one positive example per batch."
            #first shuffle the data

            self.hasDxSplits=self.clinicalDatasetSampler._indexSplit(len(self.sampling_index[0]),self.totalNumBatches)
            self.noDxSplits=self.clinicalDatasetSampler._indexSplit(len(self.sampling_index[1]),self.totalNumBatches)
            if self.totalNumBatches == 0:
                self.totalNumBatches+=1

        else:
            self.sampling_index=shuffle(self.sampling_index)
            self.totalNumBatches,leftover = divmod(len(self.sampling_index),batch_size)
            self.splits = self.clinicalDatasetSampler._indexSplit(len(self.sampling_index),self.totalNumBatches)
            if self.totalNumBatches == 0:
                self.totalNumBatches+=1

    def __len__(self):
        return self.totalNumBatches

    def shuffle_index(self):
        if self.clinicalDatasetSampler.isConditioned:
            self.sampling_index[0]=shuffle(self.sampling_index[0])
            self.sampling_index[1]=shuffle(self.sampling_index[1])
        else:
            self.sampling_index=shuffle(self.sampling_index)



    def __getitem__(self,index):
        if self.clinicalDatasetSampler.isConditioned:
            hasDxSubset = self.sampling_index[0][self.hasDxSplits[index]:self.hasDxSplits[index+1]]
            noDxSubset =  self.sampling_index[1][self.noDxSplits[index]:self.noDxSplits[index+1]]
            batchIndex = np.concatenate((hasDxSubset,noDxSubset))

        else:
            batchIndex=self.sampling_index[self.splits[index]:self.splits[index+1]]

        return self.clinicalDatasetSampler._returnData(batchIndex)
