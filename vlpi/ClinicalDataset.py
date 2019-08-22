#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:37:15 2019

@author: davidblair
"""

import numpy as np
import pandas as pd
from ICDUtilities import ICDUtilities
import torch
from scipy import sparse
from typing import Iterable

class ClinicalDataset:
    """
    Constructs a clinical dataset by reading in text files (Tab-delimited)
    and storing them as a Pandas dataset
    """
    
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
                newCodeList+=[self.codeParser(code)]
            except ValueError:
                pass
                    
        return np.array(newCodeList,dtype=np.int32)
        
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
                
                    
    def __init__(self,ICDInfo=None, isICD10 = True):
        self.isICD10=isICD10
        
        if ICDInfo ==None:
            self.ICDInfo=ICDUtilities('ICDData/CMS32_DESC_LONG_DX.txt','ICDData/icd10cm_codes_2018.txt')
        else:
            self.ICDInfo=ICDInfo
            
        if self.isICD10:
            self.codeParser = self.ICDInfo.returnICD10Int
            self.codeToIntMap = self.ICDInfo.ICD10_to_IntDict
            self.intToCodeMap = self.ICDInfo.Int_to_ICD10Dict
            self.codeToString = self.ICDInfo.ICD10Dict
            
        else:
            self.codeParser = self.ICDInfo.returnICD9Int
            self.codeToIntMap = self.ICDInfo.ICD9_to_IntDict
            self.intToCodeMap = self.ICDInfo.Int_to_ICD9Dict
            self.codeToString = self.ICDInfo.ICD9Dict
            
        self.numICDCodes = len(self.codeToIntMap)
        self.data=None
        self.numPatients = None
        self.catCovConversionDicts={}
    
    def ReadDatasetFromFile(self,clinicalDataset,dxCodeColumn,indexColumn = None, skipColumns=[], hasHeader=True,chunkSize = 500):
        """
        Initializes the Pandas dataset by reading it from file. 
        Requires ICDInfo, which is of the class ICDUtilities
        

        
        Will eventually extend to allow read/write from dataset stored only on disk.
    
            
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
        self.data = self.initialize_empty_df(colNames,[np.int32,np.object]+[np.int32 for i in range(len(catCovNames))])

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
        Finds all patients with a particular dx code, returns their IDs
        """
        
        if '.' in dx_code:
            dx_code.replace('.','')
        intVal = self.codeParser(dx_code)
        return self.data['patient_id'][self.data['dx_codes'].apply(lambda x: intVal in x)]
        
    def _updateICDInfo(self):
        if self.isICD10:
            self.ICDInfo.ICD10_to_IntDict=self.codeToIntMap 
            self.ICDInfo.Int_to_ICD10Dict=self.intToCodeMap 
            self.ICDInfo.ICD10Dict=self.codeToString 
            
        else:
            self.ICDInfo.ICD9_to_IntDict=self.codeToIntMap
            self.ICDInfo.Int_to_ICD9Dict=self.intToCodeMap 
            self.ICDInfo.ICD9Dict=self.codeToString
    
    def IncludeOnly(self,dx_code_list):
        """
        Removes all dx_codes from the dataset except those from the dx_code_list
        """
        
        dx_code_list=[x.replace('.','') for x in dx_code_list]
        allKept = set([self.codeToIntMap[x] for x in dx_code_list])
        
        #now we need to remove all non-kept codes from the ICD conversion dictionaries
        removedCodes = set(self.codeToIntMap.keys()).difference(dx_code_list)
        
        for old_code in removedCodes:
            del self.codeToIntMap[old_code]
            
        self.intToCodeMap = {}
        newCodeToIntMap = {}
        newCodeToStringMap = {}
        oldToNewIntMap={}
        for i,key in enumerate(self.codeToIntMap):
            oldToNewIntMap[self.codeToIntMap[key]]=i
            self.intToCodeMap[i]=key
            newCodeToIntMap[key] = i
            newCodeToStringMap[key] = self.codeToString[key]
            
        self.codeToIntMap=newCodeToIntMap
        self.codeToString=newCodeToStringMap
        self._updateICDInfo()
        
        try:
            self.data['dx_codes']=self.data['dx_codes'].apply(lambda x: np.array([oldToNewIntMap[y] for y in x if y in allKept],dtype=np.int32))
        except TypeError:
            pass
        self.numICDCodes=len(self.codeToIntMap)
        
    def ExcludeAll(self,dx_code_list):
        """
        Removes all codes in dx_code_list from the dataset
        
        """
        keptCodes = set(self.codeToIntMap.keys()).difference(dx_code_list)
        self.IncludeOnly(list(keptCodes))
        
    def ConditionOnDx(self,dx_code_list):
        """
        Conditions the data table on whether a patient has a particular or set of dx's
        Accomplished by finding all patients with a each dx, then adding a column (boolean)
        indicating this to be true.
        Completed by removing the codes from the data table
        """
        for dx_code in dx_code_list:
            dx_code.replace('.','')
            allPatients_wDx=self.FindAllPatients_wDx(dx_code)
            hasDx=np.zeros(self.numPatients,dtype=np.bool)
            self.data.insert(len(self.data.columns),'has_'+dx_code,hasDx)
            self.data.loc[allPatients_wDx,'has_'+dx_code]=True
        self.ExcludeAll(dx_code_list)
        
    def WriteToHDF5(self,fileName):
        """
        Writes dataset to disk, stored in hdf5 format
        """
        if fileName[-3:]!='.h5':
            fileName+='.h5'
        
        with pd.HDFStore(fileName) as dataFile:
            dataFile['data']=self.data
            for catCovName,convDict in self.catCovConversionDicts.items():
                dataFile['catCovConversionDicts/'+catCovName] = pd.DataFrame(zip(convDict.keys(),convDict.values()), columns = ['OrigVal','IntVal'])
    
        
        
        
        
    def ReadFromHDF5(self,fileName):
        
        """
        Reads dataset from hdf5 format.
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
        
        self.numPatients = len(self.data)
        
    def ReturnArrayData(self,patientIndices,arrayType="Numpy"):
        assert arrayType in ['Numpy','Torch'], "Only Numpy arrarys or Torch tensors supported"
        if arrayType =='Numpy':
            self.arrayFunc = np.array
        else:
            self.arrayFunc=torch.tensor
            
        newDF = self.data.loc[patientIndices]
        disInds = np.zeros((len(newDF),self.numICDCodes),dtype=np.float32)
        covData = [newDF[s].values for s in self.catCovConversionDicts.keys()]
        for i,onInds in enumerate(newDF['dx_codes']):
            disInds[i,onInds]=1
            
        return self.arrayFunc(disInds),[self.arrayFunc(x).reshape(len(disInds),1) for x in covData]
        
    def LoadFromArrays(self,incidenceArray,covariateArrays,covariateNames,catCovDicts=None, arrayType = 'Numpy'):
        """
        Loads clinical dataset from array data, used for simulation purposes
        """ 
        assert arrayType in ['Numpy','Torch'], "Only Numpy arrarys or Torch tensors supported"
        if covariateArrays==None:
            covariateArrays=[]
        if covariateNames==None:
            covariateNames=[]
        assert len(covariateArrays)==len(covariateNames), "Number of covariate names does not match number of covariate arrays."
        assert incidenceArray.shape[1]==self.numICDCodes, "Dimension of incidence data does not match number of codes."
        
        if arrayType=='Torch':
            incidenceArray=incidenceArray.detach().numpy()
            covariateArrays=[x.detach().numpy().ravel() for x in covariateArrays]
        else:
            covariateArrays=[x.ravel() for x in covariateArrays]
        
        dataDict={}
        for i,name in enumerate(covariateNames):
            if catCovDicts == None:  
                uniqueCats = list(set(covariateArrays[i]))
                self.catCovConversionDicts[name] = dict(zip(uniqueCats,list(range(len(uniqueCats)))))
                covariateArrays[i] = np.array([self.catCovConversionDicts[name][x] for x in covariateArrays[i]])
            else:
                self.catCovConversionDicts[name]=catCovDicts
                covariateArrays[i] = np.array([self.catCovConversionDicts[name][x] for x in covariateArrays[i]])
            dataDict[name] = covariateArrays[i]
        
        dataDict['patient_id']=np.arange(incidenceArray.shape[0],dtype=np.int64)
        dataDict['dx_codes'] = [np.array(np.where(x==1)[0],dtype=np.int32) for x in incidenceArray]
        
        self.data = pd.DataFrame(dataDict)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data.set_index('patient_id',drop=False, inplace=True)
        self.numPatients = len(self.data)
    
class ClincalDatasetSampler:
    """
    Generates random samples from a clinical dataset. Samples can be generated 
    unconditionially, or conditional on a patient having a particular dx.
    Note, that in that case, the dx will be removed from the dataset and included as
    a separate column in the data.
    
    """
    
    
    def __init__(self, currentClinicalDataset,trainingFraction,conditionSamplingOnDx:Iterable[str]=[],returnArrays='Numpy',shuffle=True):
        """
        DocString
        
        """        
        self.conditionSamplingOnDx=conditionSamplingOnDx
        self.currentClinicalDataset=currentClinicalDataset
        assert returnArrays in ['Numpy','Torch'], "Only Numpy arrarys or Torch tensors supported"
        
        if returnArrays =='Numpy':
            self.arrayFunc = np.array
        else:
            self.arrayFunc=torch.tensor
            
        if shuffle==True:
            self.currentClinicalDataset.data=self.currentClinicalDataset.data.sample(frac=1)
            
        if len(self.conditionSamplingOnDx)==0:
            self.currentClinicalDataset = currentClinicalDataset
            cutOffVal = int(np.floor(len(currentClinicalDataset.data)*trainingFraction))
            self.trainingDataIndex = currentClinicalDataset.data.index[0:cutOffVal]
            self.testDataIndex = currentClinicalDataset.data.index[cutOffVal:]
        else:
            conditionedColumns = set(['has_'+dx_code for dx_code in self.conditionSamplingOnDx])
            missingColumns  = conditionedColumns.difference(self.currentClinicalDataset.data.columns)
            
            if len(missingColumns)>0:
                self.currentClinicalDataset.ConditionOnDx([x.replace('has_','') for x in missingColumns])
            
            has_at_least_one_dx = np.sum(np.vstack([np.array(self.currentClinicalDataset.data['has_'+dx],dtype=np.bool) for dx in self.conditionSamplingOnDx]),axis=0)
            dataWithDx = self.currentClinicalDataset.data.index[has_at_least_one_dx>0]
            dataWithoutDx = self.currentClinicalDataset.data.index[has_at_least_one_dx==0]
            self.fracWDx = len(dataWithDx)/len(self.currentClinicalDataset.data)
            cutOffValWDx = int(np.floor(len(dataWithDx)*trainingFraction))
            cutOffValWoDx = int(np.floor(len(dataWithoutDx)*trainingFraction))
            
            self.trainingDataIndex=[dataWithDx[0:cutOffValWDx],dataWithoutDx[0:cutOffValWoDx]]
            self.testDataIndex=[dataWithDx[cutOffValWDx:],dataWithoutDx[cutOffValWoDx:]]

    def _createSparseDataArray(self,incArray,covArray):
        """
        Returns sparse scipy arrays instead of torch arrays
        
        incArray: array of binary dis incidence values
        covArray: integer covariates for conversion into one hot encoding
        """
        X = sparse.csr_matrix(incArray)
        for i,catDict in enumerate(self.currentClinicalDataset.catCovConversionDicts.values()):
            tmp = sparse.csr_matrix((np.ones(covArray[i].shape[0]),(np.arange(covArray[i].shape[0]),np.array(covArray[i]).ravel())),shape=(covArray[i].shape[0],len(catDict)))
            X = sparse.hstack((X,tmp))
        return X
            
    def _generateRandomSample(self,numSamples,datasetIndex,fixedFracWDx):
        if fixedFracWDx!=None:
            assert isinstance(fixedFracWDx, float) and fixedFracWDx < 1.0 and fixedFracWDx > 0.0, "fixedFrac with dx must be float between 0.0 and 1.0"
            assert len(self.conditionSamplingOnDx)>0, "Cannot include fixed fraction of positive cases if conditionSamplingOnDx not enabled"
            if fixedFracWDx*numSamples >= len(datasetIndex[0]):
                numSamplesWDx = len(datasetIndex[0])
                print("Warning: fixedFracWDx exceeds or is equal to the total number of positive training samples.\nEvery positive case will be included in random sample")
            else:
                numSamplesWDx = int(np.ceil(numSamples*fixedFracWDx))
        elif len(self.conditionSamplingOnDx)>0:
            numSamplesWDx=int(np.ceil(numSamples*self.fracWDx))
            
        disInds = np.zeros((numSamples,self.currentClinicalDataset.numICDCodes),dtype=np.float32)
        if len(self.conditionSamplingOnDx)==0:
            newDF = self.currentClinicalDataset.data.loc[np.random.choice(datasetIndex,size=numSamples,replace=False)]
            covData = [newDF[s].values for s in self.currentClinicalDataset.catCovConversionDicts.keys()]
        else:
            newDF = pd.concat((self.currentClinicalDataset.data.loc[np.random.choice(datasetIndex[0],size=numSamplesWDx,replace=False)],self.currentClinicalDataset.data.loc[np.random.choice(datasetIndex[1],size=(numSamples-numSamplesWDx),replace=False)]),ignore_index=True)
            newDF = newDF.sample(frac=1)
            covData = [newDF[s].values for s in self.currentClinicalDataset.catCovConversionDicts.keys()]
            incidenceData = np.array(pd.concat([newDF['has_'+dx] for dx in self.conditionSamplingOnDx],axis=1),dtype=np.float32)
                            
        for i,onInds in enumerate(newDF['dx_codes']):
            disInds[i,onInds]=1
            
        if len(self.conditionSamplingOnDx)==0:
            return self.arrayFunc(disInds),[self.arrayFunc(x).reshape(len(disInds),1) for x in covData]
        else:
            return self.arrayFunc(disInds),[self.arrayFunc(x).reshape(len(disInds),1) for x in covData],self.arrayFunc(incidenceData.reshape(len(disInds),len(self.conditionSamplingOnDx)))
        

        
    def GenerateRandomTrainingSample(self,numSamples, fixedFracWDx=None):
        return self._generateRandomSample(numSamples,self.trainingDataIndex,fixedFracWDx)
        
    
    def GenerateRandomTestSample(self,numSamples,fixedFracWDx=None):
        return self._generateRandomSample(numSamples,self.testDataIndex,fixedFracWDx)
        
    def _returnFullDataset(self,datasetIndex,randomize):
        if isinstance(datasetIndex,list):
            datasetIndex = np.concatenate(datasetIndex,axis=0)
            #shuffle
            dataset = self.currentClinicalDataset.data.loc[datasetIndex]
            if randomize==True:
                dataset=dataset.sample(frac=1)
        else:
            dataset = self.currentClinicalDataset.data.loc[datasetIndex]
            if randomize==True:
                dataset=dataset.sample(frac=1)
        disInds = np.zeros((len(dataset),self.currentClinicalDataset.numICDCodes),dtype=np.float32)
        covData = [dataset[s].values for s in self.currentClinicalDataset.catCovConversionDicts.keys()]
        for i,onInds in enumerate(dataset['dx_codes']):
            disInds[i,onInds]=1
        
        if len(self.conditionSamplingOnDx)==0:
            return self.arrayFunc(disInds),[self.arrayFunc(x).reshape(len(disInds),1) for x in covData]
        else:
            incidenceData = np.array(pd.concat([dataset['has_'+dx] for dx in self.conditionSamplingOnDx],axis=1),dtype=np.float32)
            return self.arrayFunc(disInds),[self.arrayFunc(x).reshape(len(disInds),1) for x in covData],self.arrayFunc(incidenceData.reshape(len(disInds),len(self.conditionSamplingOnDx)))
        
    def ReturnFullTrainingDataset(self,randomize=True):
        return self._returnFullDataset(self.trainingDataIndex,randomize)
    
    def ReturnFullTestingDataset(self,randomize=True):
        return self._returnFullDataset(self.testDataIndex,randomize)
    
    def ReturnFullTrainingDataset_Sparse(self,randomize=True):
        if len(self.conditionSamplingOnDx)==0:
            X_1,X_2 = self._returnFullDataset(self.trainingDataIndex,randomize)
            return self._createSparseDataArray(X_1,X_2)
        else:
            X_1,X_2,Y = self._returnFullDataset(self.trainingDataIndex,randomize)
            return self._createSparseDataArray(X_1,X_2),Y
        
    def ReturnFullTestingDataset_Sparse(self,randomize=True):
        if len(self.conditionSamplingOnDx)==0:
            X_1,X_2 = self._returnFullDataset(self.testDataIndex,randomize)
            return self._createSparseDataArray(X_1,X_2)
        else:
            X_1,X_2,Y = self._returnFullDataset(self.testDataIndex,randomize)
            return self._createSparseDataArray(X_1,X_2),Y
    
    def _indexSplit(self,dataSetSize,totalNumBatches):
        if totalNumBatches>0:
            nEachBatch, extras = divmod(dataSetSize, totalNumBatches)
            section_sizes = ([0] +extras * [nEachBatch+1] +(totalNumBatches-extras) * [nEachBatch])
            return np.array(section_sizes, dtype=np.int32).cumsum()
        else:
            return np.array([0]+[dataSetSize], dtype=np.int32)
        
    
    def _epoch(self,datasetIndex,batch_size):
        """
        
        """
        
        if isinstance(datasetIndex,list):
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
                hasDxSubset = self.currentClinicalDataset.data.loc[datasetIndex[0][hasDxSplits[i]:hasDxSplits[i+1]]]
                noDxSubset = self.currentClinicalDataset.data.loc[datasetIndex[1][noDxSplits[i]:noDxSplits[i+1]]]
                
                batchDataset = pd.concat((hasDxSubset,noDxSubset),axis=0)
                #need to shuffle positive cases within all others
                batchDataset=batchDataset.sample(frac=1)
                yield batchDataset
        else:
            datasetIndex=np.random.permutation(datasetIndex)
            totalNumBatches,leftover = divmod(len(datasetIndex),batch_size)
            splits = self._indexSplit(len(datasetIndex),totalNumBatches)
            if totalNumBatches == 0:
                totalNumBatches+=1
            for i in range(totalNumBatches):
                
                yield self.currentClinicalDataset.data.loc[datasetIndex[splits[i]:splits[i+1]]]
            
            
    def TrainingEpochGenerator(self, batch_size):
        for batch in self._epoch(self.trainingDataIndex,batch_size):
            disInds = np.zeros((len(batch),self.currentClinicalDataset.numICDCodes),dtype=np.float32)
            covData = [batch[s].values for s in self.currentClinicalDataset.catCovConversionDicts.keys()]
            for i,onInds in enumerate(batch['dx_codes']):
                disInds[i,onInds]=1
        
            if len(self.conditionSamplingOnDx)==0:
                yield self.arrayFunc(disInds),[self.arrayFunc(x).reshape(len(disInds),1) for x in covData]
            else:
                incidenceData = np.array(pd.concat([batch['has_'+dx] for dx in self.conditionSamplingOnDx],axis=1),dtype=np.float32)
                yield self.arrayFunc(disInds),[self.arrayFunc(x).reshape(len(disInds),1) for x in covData],self.arrayFunc(incidenceData.reshape(len(disInds),len(self.conditionSamplingOnDx)))
    
    def TestingEpochGenerator(self,batch_size):
        for batch in self._epoch(self.testDataIndex,batch_size):
            disInds = np.zeros((len(batch),self.currentClinicalDataset.numICDCodes),dtype=np.float32)
            covData = [batch[s].values for s in self.currentClinicalDataset.catCovConversionDicts.keys()]
            for i,onInds in enumerate(batch['dx_codes']):
                disInds[i,onInds]=1
        
            if len(self.conditionSamplingOnDx)==0:
                yield self.arrayFunc(disInds),[self.arrayFunc(x).reshape(len(disInds),1) for x in covData]
            else:
                incidenceData = np.array(pd.concat([batch['has_'+dx] for dx in self.conditionSamplingOnDx],axis=1),dtype=np.float32)
                yield self.arrayFunc(disInds),[self.arrayFunc(x).reshape(len(disInds),1) for x in covData],self.arrayFunc(incidenceData.reshape(len(disInds),len(self.conditionSamplingOnDx)))
        
            
        
    
    
#    
#    
