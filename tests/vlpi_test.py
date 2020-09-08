import numpy as np
import torch
import string
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.data.ClinicalDataSimulator import ClinicalDataSimulator
from vlpi.vLPI import vLPI
torch.manual_seed(1023)

## set the simulation parameters

numberOfSamples=50000
numberOfSymptoms=20

rareDiseaseFrequency=0.001
numLatentPhenotypes=2

## simulate the data
simulator = ClinicalDataSimulator(numberOfSymptoms,numLatentPhenotypes,rareDiseaseFrequency)
simulatedData=simulator.GenerateClinicalData(numberOfSamples)

## load the data into a ClinicalDataset class
clinicalData = ClinicalDataset()

#change the dataset to have only 20 symptoms (defaults to full ICD10-CM codebook), named after the first 20 letters
allICDCodes = list(clinicalData.dxCodeToDataIndexMap.keys())
symptomConversionMap=dict(zip(allICDCodes[0:numberOfSymptoms],string.ascii_uppercase[0:numberOfSymptoms]))
clinicalData.ConstructNewDataArray(symptomConversionMap)

# now load into the data structure. Note, when using in this manner, it's the users responsibility to make sure that the input data columns match the data columns of the ClinicalDataset.
clinicalData.LoadFromArrays(simulatedData['incidence_data'],simulatedData['covariate_data'],[],catCovDicts=None, arrayType = 'Torch')

## Now load the ClincalDataset into ClinicalDatasetSampler
training_data_fraction=0.75
sampler = ClinicalDatasetSampler(clinicalData,training_data_fraction,returnArrays='Torch')

## Intantation the models
infNumberOfLatentPhenotypes=10
vlpiModel= vLPI(sampler,infNumberOfLatentPhenotypes)

inference_output = vlpiModel.FitModel(batch_size=1000,errorTol=(1.0/numberOfSamples))
vlpiModel.PackageModel('ExampleModel.pth')

inferredCrypticPhenotypes=vlpiModel.ComputeEmbeddings((simulatedData['incidence_data'],simulatedData['covariate_data']))
riskFunction=vlpiModel.ReturnComponents()
