import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#Load csv and pre process
dataSet= pd.read_csv("data.csv")
genre_list = dataSet.iloc[:, -1]
genres = pd.unique(genre_list)
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#Scaler for all X between 0-1
scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(dataSet.iloc[:, :-1], dtype = float))

dict_for_df = {
    'NormalData': [],
    'PcaApplyed': []
}
#For normal data
for i in range(1,11):
    accuracy = 0
    for j in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy += accuracy_score(y_test, y_pred)
    dict_for_df['NormalData'].append(accuracy/10)

#For data which applyed pca
pca = PCA(n_components = 15)
pca.fit(X)
reduced_X = pca.transform(X)
for i in range(1,11):
    accuracy = 0
    for j in range(10):
        X_train, X_test, y_train, y_test = train_test_split(reduced_X, y, test_size=0.3)
        clf = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy += accuracy_score(y_test, y_pred)
    dict_for_df['PcaApplyed'].append(accuracy/10)



fig, ax = plt.subplots()

ind = np.arange(10) 
width = 0.35 

ax.bar(ind, dict_for_df['NormalData'], width, bottom=0, label='Normal Data')
ax.bar(ind + width, dict_for_df['PcaApplyed'], width, bottom=0, label='PCA Data')

ax.set_title('Accuracy by classifier with std ')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(ind)
ax.set_xlabel("K")
ax.set_ylabel("Accuracy")
ax.legend()

ax.autoscale_view()

plt.savefig("KNN.pdf")
plt.close()