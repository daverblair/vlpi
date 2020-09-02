import numpy as np
import torch
from vlpi.data.ClinicalDataset import ClinicalDataset,ClinicalDatasetSampler
from vlpi.data.ClinicalDataSimulator import ClinicalDataSimulator
from vlpi.vLPI import vLPI


torch.manual_seed(1023)

num_samples=50000
num_symptoms=20
training_data_fraction=0.75
validation_fraction=0.2
rare_disease_freq=0.001
sim_rank=2
inf_rank=10
isOutlier=False
sim_rank+=(int(isOutlier))

simulator = ClinicalDataSimulator(num_symptoms,sim_rank,rare_disease_freq,isOutlier=isOutlier)
simData=simulator.GenerateClinicalData(num_samples)

clinData = ClinicalDataset()
#build arbitrary list of disease codes
disList =list(clinData.dxCodeToDataIndexMap.keys())[0:num_symptoms+1]

#load data into clinical dataset
clinData.IncludeOnly(disList)
clinData.LoadFromArrays(torch.cat([simData['incidence_data'],simData['target_dis_dx'].reshape(-1,1)],axis=1),simData['covariate_data'],[],catCovDicts=None, arrayType = 'Torch')
clinData.ConditionOnDx([disList[-1]])
sampler = ClinicalDatasetSampler(clinData,training_data_fraction,returnArrays='Torch',conditionSamplingOnDx = [disList[-1]])
sampler.ConvertToUnconditional()

vlpiModel= vLPI(sampler,inf_rank)

inference_output = vlpiModel.FitModel(batch_size=1000,errorTol=(1.0/num_samples))
vlpiModel.PackageModel('ExampleModel.pth')

inferredCrypticPhenotypes=vlpiModel.ComputeEmbeddings((simData['incidence_data'],simData['covariate_data']))
riskFunction=vlpiModel.ReturnComponents()
