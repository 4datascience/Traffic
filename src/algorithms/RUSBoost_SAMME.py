import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import base

def labelToIndex(i,clf):
    return clf.labelDict[i]

def indexToLabel(i,clf):
    return clf.classes[i]

class RUSBoostClassifier_:
    
    def __init__(self,base_estimator=None,n_estimators=50,learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = [None]*n_estimators
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
        
        # `undersampling_n` elements to sample from each class, equal #samples minority class
        self.createLabelDict(np.unique(df[y_column]))
        k = len(self.classes)
        undersampling_n = min(df[y_column].value_counts())
        
        # M iterations (#WeakLearners)
        for m in range(self.n_estimators):
            
        # 1) Random UnderSampling
            df_ = pd.DataFrame()
            for label_ in self.classes:
                # Quizas muestrear usando la inversa de los pesos (.sample(weights=1/w)) para hacer overfit en los casos con menos error == underfit
                df_ = pd.concat([ df_, df[ df[y_column]==label_ ].sample(undersampling_n, replace=False) ])
            
            # Indices are dates in integer format
            w_indices_ = np.int32(df_.index.astype(np.int64)/1e9)
            w_ = np.array([w[epoch] for epoch in w_indices_])
                #/ Map from epoch in w_indices to array position in w_
            w_indices_2_w_ = {epoch: i for i, epoch in enumerate(w_indices_)}

            X_ = df_.filter(regex=(X_columns)).values
            y_ = df_[y_column].values

        # 2) WeakLearner training
            Gm = base.clone(self.base_estimator).\
                            fit(X_,y_,sample_weight=w_).predict
        
        # 3) Error-rate computation
            incorrect = Gm(X_) != y_
            errM = np.average(incorrect,weights=w_,axis=0)
            self.estimator_errors_.append(np.average(incorrect,axis=0))
        
        # 4) WeakLearner weight for ensemble computation [errM < (k-1)/k]
            BetaM = np.log((1-errM)/errM)+np.log(k-1)            
            self.models[m] = (BetaM,Gm)

        # 5) Observation weights update for next iteration with weights normalization
            w_ *= np.exp(self.learning_rate* BetaM*(incorrect*(w_ > 0)))
            norm_ = 0
            for epoch in w:
                if epoch in w_indices_:
                    w[epoch] = w_[w_indices_2_w_[epoch]]
                norm_ += w[epoch]
            for epoch in w:
                w[epoch] /= norm_
        
        self.observation_weights_ = w
        return self
            
    def createLabelDict(self,classes):
        self.labelDict = {}
        self.classes = classes
        for i,cl in enumerate(classes):
            self.labelDict[cl] = i

    def predict(self,X):
        k = len(self.classes)
        Bms_ = [Bm for Bm,Gm in self.models]
        prob_matrix = np.full((X.shape[0],k), -sum(Bms_)/(k-1))
        
        # Obtain the predicted index array with shape (#obs,#weak_learners)
        iTL = np.vectorize(labelToIndex)
        y_pred = np.stack([iTL(Gm(X),self) for Bm,Gm in self.models], axis=-1)

        # Weight the indices count using Bm associated to each weak_learner
        prob_matrix += np.apply_along_axis(lambda x: np.bincount(x, weights=Bms_, minlength=k), axis=1, arr=y_pred)*k/(k-1)
        
        iTL = np.vectorize(indexToLabel)
        return iTL(np.argmax(prob_matrix,axis=1),self)
    