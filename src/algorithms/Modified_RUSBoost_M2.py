import numpy as np
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import base

def labelToIndex(cl,clf):
    return clf.labelDict[cl]

def indexToLabel(i,clf):
    return clf.classes[i]

class ModifiedRUSBoostClassifier_:
    
    def __init__(self,base_estimator=None,n_estimators=50,learning_rate=1.0):
        print("M2 Implementation of MODIFIED RUSBoost")
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
        
        # Class labels mapping to indices
        self.createLabelDict(np.unique(df[y_column]))
        k = len(self.classes)
        # `undersampling_n` elements to sample from each class, equal #samples minority class
        undersampling_n = min(df[y_column].value_counts())

        # Initialize observation weights as 1/(N*(k-1)) where N is total `n_samples` and k is the number of classes
        N = df.shape[0]
        B = N*(k-1)
        D = {epoch: np.full(k-1,1/B) for epoch in df.index}

        # Get whole dataset samples to later calculate the weighting factors on every iteration
        X = df.filter(regex=(X_columns)).values
        y = df[y_column].values
        iTL = np.vectorize(labelToIndex)
        y_indices = iTL(y,self)
        
        # M iterations (#WeakLearners)
        for m in range(self.n_estimators):

        # 1) Random UnderSampling
            df_ = pd.DataFrame()
            for label_ in self.classes:
                _df_label_ = df[ df[y_column]==label_ ]
                if len(_df_label_) <= undersampling_n:
                    df_ = pd.concat([ df_, df[ df[y_column]==label_ ] ])
                else:
                    # Order weights array in decreasing order to select the first `undersampling_n` elements
                    _D_ = {epoch: weight for epoch, weight in D.items() if epoch in _df_label_.index}
                    _D_ = [epoch for epoch,_ in sorted(_D_.items(), key = lambda x: sum(itemgetter(1)(x)), reverse = True)[:undersampling_n]]
                    df_ = pd.concat([ df_, _df_label_.loc[_D_] ])

            
            # Training data initalization
            X_ = df_.filter(regex=(X_columns)).values
            y_ = df_[y_column].values
            D_ = np.sum([D[epoch] for epoch in df_.index], axis=-1)

        # 2) WeakLearner training
            Gm = base.clone(self.base_estimator).\
                            fit(X_,y_,sample_weight=D_).predict_proba
            self.models.append(Gm)
        
        # 3) Error-rate computation
            predictions_proba = Gm(X)
            sum_pseudolosses = 0
            for i, epoch in enumerate(D.keys()):
                k_index = 0
                for cl in range(k):
                    if cl != y_indices[i]:
                        sum_pseudolosses += D[epoch][k_index]*(1-predictions_proba[i,y_indices[i]]+predictions_proba[i,cl])
                        k_index += 1

            error = 0.5 * sum_pseudolosses
            self.estimator_errors_.append(error)
        
        # 4) WeakLearner weight for ensemble computation
            BetaM = error/(1-error)
            self.models[m] = (BetaM,Gm)

        # 5) Observation weights update for next iteration with weights normalization
            norm_ = 0
            for i, epoch in enumerate(D.keys()):
                k_index = 0
                for cl in range(k):
                    if cl != y_indices[i]:
                        w_ = 0.5*(1+predictions_proba[i,y_indices[i]]-predictions_proba[i,cl])
                        D[epoch][k_index] *= BetaM**(self.learning_rate*w_)
                        norm_ += D[epoch][k_index]
                        k_index += 1
            for epoch in D.keys():
                D[epoch] /= norm_

        self.observation_weights_ = D
        
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