from sklearn.decomposition import NMF,FastICA
from sklearn.linear_model import LinearRegression
import scipy.sparse as sprs
import numpy as np

class PhenotypeDecomposition():
            
    def _buildCovariateEffectsModel(self,scores,covariateMatrix):
        lin_mod=LinearRegression()
        lin_mod.fit(covariateMatrix,scores)
        return lin_mod   


    def _build_initial_matrix(self,data_matrix):
        ica = FastICA(self.num_components,tol=1e-4)
        W_mat = np.abs(ica.fit_transform(data_matrix))
        H_mat = np.abs(ica.mixing_.T)

        return W_mat,H_mat
        
    
    def __init__(self,num_components, **kwargs):
        """
        linear matrix embedding-based anomaly detection, takes scipy.sparse arrays as input
        """
        allKeywordArgs = list(kwargs.keys())
        
        self.neg_log_like_on=None
        self.neg_log_like_off=None
        self.phenotype_centroid=None
        self.num_components=num_components
        
        if 'max_iter' not in allKeywordArgs:
            max_iter=500
        else:
            max_iter=kwargs['max_iter']
        
        if 'alpha' not in allKeywordArgs:
            alpha=0.0
        else:
            alpha=kwargs['alpha']
            
        if 'l1_ratio' not in allKeywordArgs:
            l1_ratio=0.0
        else:
            l1_ratio=kwargs['l1_ratio']
            
        if 'error_tol' not in allKeywordArgs:
            error_tol=1e-4
        else:
            error_tol=kwargs['error_tol']
            
        self.embed_model = NMF(n_components=num_components,init='custom',solver='mu',tol=error_tol,beta_loss=1,verbose=False,max_iter=max_iter,alpha=alpha,l1_ratio=l1_ratio)
        
    def _transform_data_matrix(self,data_matrix):
        new_data_matrix=np.zeros(data_matrix.shape)
        new_data_matrix+=data_matrix.toarray()*self.neg_log_like_on
        new_data_matrix+=(1-data_matrix.toarray())*self.neg_log_like_off
        return new_data_matrix
    
    def FitModel(self,training_data,covariateMatrix=None):
        
        if sprs.issparse(training_data)==False:
            training_data=sprs.csr_matrix(training_data)
            
        disPrevVec = training_data.sum(axis=0)/training_data.shape[0]
        disPrevVec[disPrevVec==0.0]=np.finfo(np.double).resolution
        disPrevVec[disPrevVec==1.0]=1.0-np.finfo(np.double).resolution
        disPrevVec=np.array(disPrevVec)
        
        self.neg_log_like_on=-1.0*np.log(disPrevVec)
        self.neg_log_like_off=-1.0*np.log(1.0-disPrevVec)
            
        transformed_data=self._transform_data_matrix(training_data)
        
        W_init,H_init=self._build_initial_matrix(transformed_data)
            
        training_data_scores=self.embed_model.fit_transform(transformed_data,W=W_init,H=H_init)
        
        
        if covariateMatrix is not None:
            if sprs.issparse(covariateMatrix)==False:
                covariateMatrix=sprs.csr_matrix(covariateMatrix)
            self.covariatesIncluded=True            
            self.linearCovariateModel=self._buildCovariateEffectsModel(training_data_scores,covariateMatrix)
            training_data_scores-=self.linearCovariateModel.predict(covariateMatrix)
        else:
            self.linearCovariateModel=None
            self.covariatesIncluded=False
            
        self.phenotype_centroid = np.mean(training_data_scores,axis=0)
        return training_data_scores,self.phenotype_centroid
        

    def ComputeScores(self,data_array,covariate_matrix=None):
        if sprs.issparse(data_array)==False:
            data_array=sprs.csr_matrix(data_array)
            
        transformed_data=self._transform_data_matrix(data_array)
        embeddings=self.embed_model.transform(transformed_data)
            
        if self.covariatesIncluded==True:
            assert covariate_matrix is not None, "Model trained with covariate matrix. Must be provided."
            if sprs.issparse(data_array)==False:
                covariate_matrix=sprs.csr_matrix(covariate_matrix)
            embeddings-=self.linearCovariateModel.predict(covariate_matrix)
        euclid_dist_from_centroid = np.sqrt(np.sum((embeddings-self.phenotype_centroid)**2.0,axis=1,keepdims=True))
        return embeddings,euclid_dist_from_centroid
        
    def ReturnComponents(self):
        return self.embed_model.components_

    

class PheRS:
    
    def __init__(self):
        self.logIDF=None
        
    def _buildCovariateEffectsModel(self,scores,covariateMatrix):
        lin_mod=LinearRegression()
        lin_mod.fit(covariateMatrix,scores)
        return lin_mod
    
    def FitModel(self,trainingData,covariateMatrix=None):
        if sprs.issparse(trainingData)==False:
            trainingData=sprs.csr_matrix(trainingData)
        disPrevVec = trainingData.sum(axis=0)/trainingData.shape[0]
        disPrevVec[disPrevVec==0.0]=1.0/(2.0*trainingData.shape[0])
        disPrevVec[disPrevVec==1.0]=1.0-(1.0/(2.0*trainingData.shape[0]))
        self.logIDF=-1.0*np.log(disPrevVec)
        training_data_scores=trainingData.dot(self.logIDF.T)
        if covariateMatrix is not None:
            if sprs.issparse(covariateMatrix)==False:
                covariateMatrix=sprs.csr_matrix(covariateMatrix)
            self.covariatesIncluded=True
            self.linearCovariateModel=self._buildCovariateEffectsModel(training_data_scores,covariateMatrix)
            training_data_scores-=self.linearCovariateModel.predict(covariateMatrix)
        else:
            self.linearCovariateModel=None
            self.covariatesIncluded=False
        return training_data_scores
    
        
        
    def ComputeScores(self,data_array,covariate_matrix=None):
        if sprs.issparse(data_array)==False:
            data_array=sprs.csr_matrix(data_array)
            
        scores=np.array(data_array.dot(self.logIDF.T))
            
        if self.covariatesIncluded==True:
            assert covariate_matrix is not None, "Model trained with covariate matrix. Must be provided."
            if sprs.issparse(data_array)==False:
                covariate_matrix=sprs.csr_matrix(covariate_matrix)
            scores-=self.linearCovariateModel.predict(covariate_matrix)
        return scores
        

class JaccardSimilarity:
    
    def _buildCovariateEffectsModel(self,scores,covariateMatrix):
        lin_mod=LinearRegression()
        lin_mod.fit(covariateMatrix,scores)
        return lin_mod   
    
    def __init__(self,index_vec,weighted=True):
        self.index_vec=index_vec
        if len(self.index_vec.shape)==1:
            self.index_vec=self.index_vec.reshape(1,self.index_vec.shape[0])
        self.weighted=weighted
        
    def FitModel(self,trainingData,covariateMatrix=None):
        if sprs.issparse(trainingData)==False:
            trainingData=sprs.csr_matrix(trainingData)
        
        if self.weighted:
            disPrevVec = trainingData.sum(axis=0)/trainingData.shape[0]
            disPrevVec[disPrevVec==0.0]=1.0/(2.0*trainingData.shape[0])
            disPrevVec[disPrevVec==1.0]=1.0-(1.0/(2.0*trainingData.shape[0]))
            self.logIDF=-1.0*np.log(disPrevVec)
        else:
            self.logIDF=np.matrix(np.ones((1,trainingData.shape[1])))
            
        trainingData=sprs.csr_matrix(trainingData.multiply(self.logIDF))
        
        intersect = (trainingData.multiply(self.index_vec)).sum(axis=1)
        union = (trainingData.sum(axis=1)+(self.logIDF*self.index_vec.T))-intersect
        training_data_scores=intersect/union
        training_data_scores[np.isnan(training_data_scores)]=0.0
        
        if covariateMatrix is not None:
            if sprs.issparse(covariateMatrix)==False:
                covariateMatrix=sprs.csr_matrix(covariateMatrix)
            self.covariatesIncluded=True
            self.linearCovariateModel=self._buildCovariateEffectsModel(training_data_scores,covariateMatrix)
            training_data_scores-=self.linearCovariateModel.predict(covariateMatrix)
        else:
            self.linearCovariateModel=None
            self.covariatesIncluded=False
        return training_data_scores
        
    
        
    def ComputeScores(self,data_array,covariateMatrix=None):
        if sprs.issparse(data_array)==False:
            data_array=sprs.csr_matrix(data_array)
        data_array=sprs.csr_matrix(data_array.multiply(self.logIDF))
            
        intersect = (data_array.multiply(self.index_vec)).sum(axis=1)
        union = (data_array.sum(axis=1)+(self.logIDF*self.index_vec.T))-intersect
        scores=intersect/union
        scores[np.isnan(scores)]=0.0
        if self.covariatesIncluded==True:
            assert covariateMatrix is not None, "Model trained with covariate matrix. Must be provided."
            if sprs.issparse(data_array)==False:
                covariateMatrix=sprs.csr_matrix(covariateMatrix)
            scores-=self.linearCovariateModel.predict(covariateMatrix)
        return scores
            
if __name__=='__main__':
    from sklearn.metrics import precision_recall_curve,average_precision_score
    import matplotlib.pyplot as plt
    
    
    nLatentDim=4
    nFitDim=4
    numMendelianComponents=2
    nObsDis=20
    nSamples=100000
    outlierFrac = 0.001
    from vlpi.ClinicalDataSimulator import ClinicalDataSimulator
    
    simulator = ClinicalDataSimulator(nObsDis,nLatentDim,latentPhenotypeEffectsPrior=[1.0,5.0],anchorDxNoisePrior=[0.5,10.0],interceptPriors=[-3.0,1.0],anchorDxThresholdPrior=[0.0001,0.01],numMendelianComponents=numMendelianComponents)
    simData=simulator.GenerateClinicalData(nSamples,outlierFrac)
    simAnchors = simulator.GenerateAnchoringDx(simData['latent_phenotypes'])
    
    phers=PheRS()
    phers_scores=phers.FitModel(simData['incidence_data'].numpy())
    
    pr=precision_recall_curve(simAnchors['anchor_dx_data'].numpy(),phers_scores)
    plt.step(pr[1],pr[0],color='b')
    
    nmf=PhenotypeDecomposition(nFitDim)
    scores,centroid=nmf.FitModel(simData['incidence_data'].numpy())
    scores,dist = nmf.ComputeScores(simData['incidence_data'].numpy())
    phenotype_components = nmf.ReturnComponents()
    
    pr=precision_recall_curve(simAnchors['anchor_dx_data'].numpy(),dist)
    plt.step(pr[1],pr[0],color='r')
    

    pred_anchors_true = np.sum(simAnchors['model_params']['anchorDxMap'].numpy()*simData['latent_phenotypes'].numpy(),axis=1,keepdims=True)
    pr=precision_recall_curve(simAnchors['anchor_dx_data'].numpy(),pred_anchors_true)
    plt.step(pr[1],pr[0],color='g')
    
    plt.figure()
    centroid=simData['latent_phenotypes'].numpy().mean(axis=0)
    true_euclid_dist = np.sqrt(np.sum((simData['latent_phenotypes'].numpy()-centroid)**2,axis=1,keepdims=True))
    
    pred_disease_liability = np.sum(simAnchors['model_params']['anchorDxMap'].numpy()*simData['latent_phenotypes'].numpy(),axis=1,keepdims=True)
    
    plt.plot(true_euclid_dist[simData['is_outlier'].numpy()==0],pred_disease_liability[simData['is_outlier'].numpy()==0],'o')
    plt.plot(true_euclid_dist[simData['is_outlier'].numpy()==1],pred_disease_liability[simData['is_outlier'].numpy()==1],'o')
    
    plt.figure()
    
    plt.plot(true_euclid_dist[simData['is_outlier'].numpy()==0],dist[simData['is_outlier'].numpy()==0],'o')
    plt.plot(true_euclid_dist[simData['is_outlier'].numpy()==1],dist[simData['is_outlier'].numpy()==1],'o')
    
    plt.figure()
    plt.plot(true_euclid_dist[simAnchors['anchor_dx_data'].numpy()==0],pred_disease_liability[simAnchors['anchor_dx_data'].numpy()==0],'o')
    plt.plot(true_euclid_dist[simAnchors['anchor_dx_data'].numpy()==1],pred_disease_liability[simAnchors['anchor_dx_data'].numpy()==1],'o')
    
#
    
    