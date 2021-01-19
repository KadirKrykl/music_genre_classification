#Libraries

#Create and read csv files
import os
import csv
import librosa #to create musin info for csv file

import pandas as pd #to process data from csv
import numpy as np #to process data from csv

#For machine learning preprocess
import sklearn as skl
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#Machine Learning Algorithms
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


class Classifier():
    genre_list=""
    genres=""
    X=""
    y=""
    clf=""
    def __init__(self, classifier, bagging, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        dataSet= pd.read_csv("data.csv")
        self.genre_list = dataSet.iloc[:, -1]
        self.genres = pd.unique(self.genre_list)
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.genre_list)
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(np.array(dataSet.iloc[:, :-1], dtype = float))
        if(bagging):
            if(classifier == "KNN"):
                self.clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_estimators=4)
            elif(classifier == "SVM"):
                self.clf = BaggingClassifier(SVC(gamma='auto'), n_estimators=4)
            elif(classifier == "MLP"):
                self.clf = BaggingClassifier(MLPClassifier(hidden_layer_sizes=(256,128,64,8), max_iter=1000), n_estimators=4)
        else:
            if(classifier == "KNN"):
                self.clf = KNeighborsClassifier(n_neighbors=5)
            elif(classifier == "SVM"):
                self.clf = SVC(gamma='auto')
            elif(classifier == "MLP"):
                self.clf = MLPClassifier(hidden_layer_sizes=(256,128,64,8), max_iter=1000)
        print("Classifier init")

    def train(self):
        self.clf.fit(self.X, self.y)

    def predict(self,filePath):
        test=[]
        y, sr = librosa.load(filePath, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        test.append(np.mean(chroma_stft))
        test.append(np.mean(spec_cent))
        test.append(np.mean(spec_bw))
        test.append(np.mean(rolloff))
        test.append(np.mean(zcr))
        for e in mfcc:
            test.append(np.mean(e))
        
        y_prob = self.clf.predict_proba([test])
        y_prob = y_prob.tolist()[0]
        y_prob2 = [float("{:.3f}".format(float(i))) for i in y_prob]
        y_prob2 = [(i/sum(y_prob2))*100 for i in y_prob2]
        y_prob2 = {self.genres[i]:float("{:.2f}".format(float(y_prob2[i]))) for i in range(len(y_prob2))}
        return y_prob2
