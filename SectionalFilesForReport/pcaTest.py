import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

#Load csv and pre process for pca
dataSet= pd.read_csv("data.csv")
genre_list = dataSet.iloc[:, -1]
genres = pd.unique(genre_list)
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#Scaler for all X between 0-1
scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(dataSet.iloc[:, :-1], dtype = float))

pca = PCA(n_components = 0.95)
pca.fit(X)
reduced_X = pca.transform(X)

#Ploting

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, pca.explained_variance_ratio_.shape[0]+1, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, pca.explained_variance_ratio_.shape[0]+1, step=1)) 
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.savefig("Pca_Test.pdf")
plt.close()