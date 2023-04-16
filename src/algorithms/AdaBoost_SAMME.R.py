import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import base

def labelToIndex(cl,clf):
    return clf.labelDict[cl]

def indexToLabel(i,clf):
    return clf.classes[i]

class AdaBoostClassifier_:
    
    def __init__(self,base_estimator=None,n_estimators=50,learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        if base_estimator == None:
            base_estimator = DecisionTreeClassifier(max_depth=1)
        self.base_estimator = base_estimator
        self.estimator_errors_ = []
        self.observation_weights_ = {}
        
    def fit(self, df: pd.DataFrame, X_columns: str, y_column: str):
        """
        param df: Training dataframe with shape (n_samples, )
        param X_columns: pattern matching DataFrame column names that contain the features
        param y_column: name of DataFrame column name with the labels
        """
        
        # Initialize observation weights as 1/N where N is total `n_samples`
        N = df.shape[0]
        w = {epoch: 1/N for epoch in np.int32(df.index.astype(np.int64)/1e9)}
        
        # Class labels mapping to indices
        self.createLabelDict(np.unique(df[y_column]))
        k = len(self.classes)

        # Training data initalization
        X_ = df.filter(regex=(X_columns)).values
        y_ = df[y_column].values
        w_indices_ = np.int32(df.index.astype(np.int64)/1e9)
        
        # M iterations (#WeakLearners)
        for m in range(self.n_estimators):
            w_ = np.array([w[i] for i in w_indices_])

        # 1) WeakLearner training
            Gm = base.clone(self.base_estimator).\
                            fit(X_,y_,sample_weight=w_).predict_proba
            self.models.append(Gm)
        
        # 2) Error-rate computation
            sum_model_hypothesis = np.sum(np.stack([self._prob2classWeight_(model(X_)) for model in self.models], axis=-1), axis=-1)
            iTL = np.vectorize(indexToLabel)            
            incorrect = iTL(np.argmax(sum_model_hypothesis),self) != y_
            self.estimator_errors_.append(np.average(incorrect,axis=0))

        # 3) Observation weights update for next iteration with weights normalization
            iTL = np.vectorize(labelToIndex)
            tmp_ = iTL(y_,self)
            w_ *= np.exp(-self.learning_rate* (k-1)/k * np.log(Gm(X_)[np.arange(len(tmp_)),tmp_]))
            norm_ = sum(w_)
            for i, j in enumerate(w_indices_):
                w[j] = w_[i]/norm_
        
        self.observation_weights_ = w
        return self
            
    def createLabelDict(self,classes):
        self.labelDict = {}
        self.classes = classes
        for i,cl in enumerate(classes):
            self.labelDict[cl] = i

    def _prob2classWeight_(self, probabilities):
        """
        Following the SAMME.R algorithm, returns the hypothesis probabilities for each class
        param probabilities: The class probabiilties output by the fitted model (n_samples, n_classes)
        """
        k = len(self.classes)
        tmp_ = np.log(probabilities)
        h_k = np.log(probabilities)
        for i in range(k):
            h_k[:,i] -= np.sum(np.delete(tmp_,i,axis=1), axis=1) / k
        h_k *= (k-1)
        return h_k
    
    def predict(self,X):
        sum_model_hypothesis = np.sum(np.stack([self._prob2classWeight_(model(X)) for model in self.models], axis=-1), axis=-1)
        iTL = np.vectorize(indexToLabel)            
        return iTL(np.argmax(sum_model_hypothesis),self)
