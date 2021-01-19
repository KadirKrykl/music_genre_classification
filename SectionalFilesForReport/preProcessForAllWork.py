import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#Load csv and pre process
dataSet= pd.read_csv("data.csv")
genre_list = dataSet.iloc[:, -1]
genres = pd.unique(genre_list)
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#Scaler for all X between 0-1
scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(dataSet.iloc[:, :-1], dtype = float))