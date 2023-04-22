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
        print("M2 Implementation of AdaBoost")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        if base_estimator == None:
            base_estimator = DecisionTreeClassifier(max_depth=1)
        self.base_estimator = base_estimator
        self.estimator_errors_ = []

    def fit(self, df: pd.DataFrame, X_columns: str, y_column: str):
        """
        param df: Training dataframe with shape (n_samples, )
        param X_columns: pattern matching DataFrame column names that contain the features
        param y_column: name of DataFrame column name with the labels
        """        
        
        # Class labels mapping to indices
        self.createLabelDict(np.unique(df[y_column]))
        k = len(self.classes)

        # Initialize observation weights as 1/(N*(k-1)) where N is total `n_samples` and k is the numebr of classes
        N = df.shape[0]
        B = N*(k-1)
        D = {epoch: [1/B]*(k-1) for epoch in df.index}

        # Training data initalization
        X_ = df.filter(regex=(X_columns)).values
        y_ = df[y_column].values
        
        # M iterations (#WeakLearners)
        for m in range(self.n_estimators):
            D_ = np.sum(list(D.values()), axis=-1)
            iTL = np.vectorize(labelToIndex)
            y_indices_ = iTL(y_,self)

        # 1) WeakLearner training
            Gm = base.clone(self.base_estimator).\
                            fit(X_,y_,sample_weight=D_).predict_proba
            self.models.append(Gm)
        
        # 2) Error-rate computation
            predictions_proba = Gm(X_)
            sum_pseudolosses = 0
            for i, epoch in enumerate(D):
                k_index = 0
                for cl in range(k):
                    if cl != y_indices_[i]:
                        sum_pseudolosses += D[epoch][k_index]*(1-predictions_proba[i,y_indices_[i]]+predictions_proba[i,cl])
                        k_index += 1

            error = 0.5 * sum_pseudolosses
            self.estimator_errors_.append(error)
        
        # 3) WeakLearner weight for ensemble computation
            BetaM = error/(1- error +1e-8)
            self.models[m] = (BetaM,Gm)

        # 4) Observation weights update for next iteration with weights normalization
            norm_ = 0
            for i, epoch in enumerate(D.keys()):
                k_index = 0
                for cl in range(k):
                    if cl != y_indices_[i]:
                        w_ = 0.5*(1+predictions_proba[i,y_indices_[i]]-predictions_proba[i,cl])
                        D[epoch][k_index] *= BetaM**(self.learning_rate*w_)
                        norm_ += D[epoch][k_index]
                        k_index += 1
            for epoch in D.keys():
                for k_index in range(k-1):
                    D[epoch][k_index] /= norm_
        
        return self
            
    def createLabelDict(self,classes):
        self.labelDict = {}
        self.classes = classes
        for i,cl in enumerate(classes):
            self.labelDict[cl] = i
    
    def predict(self,X):
        sum_model_hypothesis = np.sum(np.stack([-np.log(Bm)*Gm(X) for Bm,Gm in self.models], axis=-1), axis=-1)
        iTL = np.vectorize(indexToLabel)            
        return iTL(np.argmax(sum_model_hypothesis,axis=1),self)
