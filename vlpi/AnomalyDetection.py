from sklearn.decomposition import TruncatedSVD,NMF
import scipy.sparse as sprs
from scipy import stats
import numpy as np

class EmbeddingAnomalyDetection():

    def __init__(self,trainingData,testingData,num_components=1,embedding_model='NMF', **kwargs):
        """
        PCA-based anomaly detection, takes scipy.sparse arrays as input
        """
        assert embedding_model in ['PCA','NMF'],'Only PCA and NMF supported'
        if sprs.issparse(trainingData)==False:
            trainingData=sprs.csr_matrix(trainingData)
        if sprs.issparse(testingData)==False:
            testingData=sprs.csr_matrix(testingData)
        
        self.trainingData=trainingData
        self.testingData=testingData
        if embedding_model=='NMF':
            self.embed_model = NMF(n_components=num_components,init='nndsvd',solver='cd')
        else:
            self.embed_model = TruncatedSVD(n_components=num_components)

    def ComputeEmbeddings(self,n_components):
        encodedTrainingData = self.embed_model.fit_transform(self.trainingData)
        encodedTestingData=self.embed_model.transform(self.testingData)
        return encodedTrainingData,encodedTestingData

    def ComputeReconstructionScores(self,n_components):
        encodedTrainingData = self.embed_model.fit_transform(self.trainingData)
        encodedTestingData=self.embed_model.transform(self.testingData)
        decodedTrainingData = self.embed_model.inverse_transform(encodedTrainingData)
        decodedTestingData = self.embed_model.inverse_transform(encodedTestingData)

        sqErrorTrain = np.sum((decodedTrainingData-self.trainingData.toarray())**2,axis=1,keepdims=True)
        sqErrorTest = np.sum((decodedTestingData-self.testingData.toarray())**2,axis=1,keepdims=True)

        #transform to z-scores
        trainScores = stats.zscore(sqErrorTrain,ddof=1)
        testScores = stats.zscore(sqErrorTest,ddof=1)
        return trainScores,testScores

class PheRS_AnomalyDetection:
    
    def __init__(self, trainingData, testingData):
        if sprs.issparse(trainingData)==False:
            trainingData=sprs.csr_matrix(trainingData)
        if sprs.issparse(testingData)==False:
            testingData=sprs.csr_matrix(testingData)
        self.trainingData=trainingData
        self.testingData=testingData
        
        disPrevVec = trainingData.sum(axis=0)/trainingData.shape[0]
        disPrevVec[disPrevVec==0.0]=1.0/(2.0*trainingData.shape[0])
        self.logIDF=-1.0*np.log(disPrevVec)
        
    def ComputeScores(self):
        training_scores = np.array(self.trainingData.dot(self.logIDF.T))
        testing_scores = np.array(self.testingData.dot(self.logIDF.T))
        return training_scores,testing_scores

class JaccardSimilarity:
    
    def __init__(self,trainingData,testingData,index_vec):
        if sprs.issparse(trainingData)==False:
            trainingData=sprs.csr_matrix(trainingData)
        else:
            testingData=sprs.csr_matrix(testingData)
            
        self.trainingData=trainingData
        self.testingData=testingData
        self.index_vec=index_vec
        
    def ComputeScores(self):
        intersect_training = (self.trainingData.multiply(self.index_vec)).sum(axis=1)
        union_training = (self.trainingData.sum(axis=1)+self.index_vec.sum())-intersect_training
        
        
        intersect_testing =  (self.testingData.multiply(self.index_vec)).sum(axis=1)
        union_testing = (self.testingData.sum(axis=1)+self.index_vec.sum())-intersect_testing
        
        output_training=intersect_training/union_training
        output_testing = intersect_testing/union_testing
        #handle division by zero
        output_training[np.isnan(output_training)]=0.0
        output_testing[np.isnan(output_testing)]=0.0
        return output_training,output_testing
            
            
