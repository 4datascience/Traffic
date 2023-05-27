import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from tensorflow import keras

def labelToIndex(cl,clf):
    return clf.labelDict[cl]

def indexToLabel(i,clf):
    return clf.classes[i]

class LSTMRUSBoostClassifier_:
    
    def __init__(self, sequence_length: int, n_lstm=50, learning_rate=1.0):
        """
        param sequence_length: Length of each feature sequence. sequence_length = N in the feature sample [x(t),...,x(t-N)]
        """
        print("M2 Implementation of LSTM RUSBoost")
        self.n_lstm = n_lstm
        self.learning_rate = learning_rate
        self.models = []
        self.scaler = []
        self.onehot_encoder = preprocessing.OneHotEncoder
        self.sequence_length = sequence_length
        self.estimator_errors_ = []
        self.observation_weights_ = {}

    def fit(self, df: pd.DataFrame, X_columns: str, y_column: str):
        """
        param df: Training dataframe with shape (n_samples, ). Sample sequence order is expected to be received as [x(t),...,x(t-N)]
        param X_columns: pattern matching DataFrame column names that contain the features
        param y_column: name of DataFrame column name with the labels
        """        
        
        # Class labels mapping to internal indices ('lab1'->0, 'lab2'->1,...'labN'->N)
        self.createLabelDict(np.unique(df[y_column]))
        # `undersampling_n` elements to sample from each class, equal #samples minority class
        undersampling_n = min(df[y_column].value_counts())

        # Initialize observation weights as 1/(N*(k-1)) where N is total `n_samples` and k is the number of classes
        N = df.shape[0]
        k = len(self.classes)
        B = N*(k-1)
        D = {epoch: np.full(k-1,1/B) for epoch in df.index}

        # Get whole dataset samples to later calculate the weighting factors on every iteration
        X = df.filter(regex=(X_columns)).values
        if X.shape[1] == self.sequence_length:
            X = np.expand_dims(X, axis=2)
        elif X.shape[1] % self.sequence_length == 0:
            X = X.reshape((X.shape[0],self.sequence_length,int(X.shape[0]/self.sequence_length)), order='F')
        else:
            print("The length of all feature sequences must be the same")
        #Correctly order the sequence [X(t-N),...,X(t)]
        X = np.flip(X, 1)

        # Initialize input scaler for each feature
        for feature in range(X.shape[2]):
            self.scaler.append( preprocessing.MinMaxScaler().fit(np.array([[X[:,-1,feature].max()]*self.sequence_length, [X[:,-1,feature].min()]*self.sequence_length])) )
            X[:,:,feature] = self.scaler[feature].transform(X[:,:,feature])
        
        # Initialize output one-hot vector encoder
        y = df[y_column].values
        iTL = np.vectorize(labelToIndex)
        y_indices = iTL(y,self)
        self.onehot_encoder = self.onehot_encoder().fit(np.expand_dims(y, axis=1))
        
        # M iterations (#WeakLearners)
        for m in range(self.n_lstm):

        # 1) Random UnderSampling
            df_ = pd.DataFrame()
            for label_ in self.classes:
                df_ = pd.concat([ df_, df[ df[y_column]==label_ ].sample(undersampling_n, replace=False) ])
            
            # Training data initalization
            X_ = df_.filter(regex=(X_columns)).values
            if X_.shape[1] == self.sequence_length:
                X_ = np.expand_dims(X_, axis=2)
            elif X_.shape[1] % self.sequence_length == 0:
                X_ = X_.reshape((X_.shape[0],self.sequence_length,int(X_.shape[0]/self.sequence_length)), order='F')
            # Scale inputs
            for feature in range(X_.shape[2]):
                X_[:,:,feature] = self.scaler[feature].transform(X_[:,:,feature])
            #Correctly order the sequence [X(t-N),...,X(t)]
            X_ = np.flip(X_, 1)

            y_ = df_[y_column].values
            y_ = self.onehot_encoder.transform(np.expand_dims(y_, axis=1)).toarray()
            D_ = np.sum([D[epoch] for epoch in df_.index], axis=-1)

        # 2) LSTM training
            num_features = X_.shape[2]
            num_out = y_.shape[1]

            model = keras.models.Sequential()

            model.add(keras.layers.LSTM(
                    input_shape=(self.sequence_length, num_features),
                    units=32,
                    return_sequences=False))
            model.add(keras.layers.Dropout(0.2))

            model.add(keras.layers.Dense(units=num_out, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_, y_, sample_weight=D_, epochs=5, batch_size=32, validation_split=0, verbose=0)
        
        # 3) Error-rate computation
            predictions_proba = model.predict(X,verbose=None)
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
            self.models.append( (BetaM,model.predict) )

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
        if X.shape[1] == self.sequence_length:
            X = np.expand_dims(X, axis=2)
        elif X.shape[1] % self.sequence_length == 0:
            X = X.reshape((X.shape[0],self.sequence_length,int(X.shape[0]/self.sequence_length)), order='F')
        else:
            print("The length of all feature sequences must be the same")
        
        # Scale inputs
        for feature in range(X.shape[2]):
            X[:,:,feature] = self.scaler[feature].transform(X[:,:,feature])
        
        sum_model_hypothesis = np.sum(np.stack([-np.log(Bm)*Gm(X) for Bm,Gm in self.models], axis=-1), axis=-1)
        return self.onehot_encoder.inverse_transform(sum_model_hypothesis)