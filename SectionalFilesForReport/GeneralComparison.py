import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

N = 10

#Load csv and pre process
dataSet= pd.read_csv("data.csv")
genre_list = dataSet.iloc[:, -1]
genres = pd.unique(genre_list)
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#Scaler for all X between 0-1
scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(dataSet.iloc[:, :-1], dtype = float))

#For data which applyed pca
pca = PCA(n_components = 15)
pca.fit(X)
reduced_X = pca.transform(X)

classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    SVC(gamma='auto'),
    MLPClassifier(hidden_layer_sizes=(128,64,8), max_iter=1000),
    BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_estimators=4),
    BaggingClassifier(SVC(gamma='auto'), n_estimators=4),
    BaggingClassifier(MLPClassifier(hidden_layer_sizes=(128,64,8), max_iter=1000), n_estimators=4)
]
normal_data = {
    'KNN': [],
    'SVM': [],
    'MLP': [],
    'Bag_KNN': [],
    'Bag_SVM': [],
    'Bag_MLP': [],
}

for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for classifier,name in zip(classifiers,normal_data.keys()):
        print("Classsifier:{0}, i:{1}".format(name,i))
        clf = classifier.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        normal_data[name].append(accuracy_score(y_test, y_pred))

normal_data_std = [np.array(value).std() for value in normal_data.values()]
normal_data_avg = [np.array(value).mean() for value in normal_data.values()]

pca_data = {
    'KNN': [],
    'SVM': [],
    'MLP': [],
    'Bag_KNN': [],
    'Bag_SVM': [],
    'Bag_MLP': [],
}


for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    for classifier,name in zip(classifiers,normal_data.keys()):
        print("Classsifier:{0}, i:{1}".format(name,i))
        clf = classifier.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pca_data[name].append(accuracy_score(y_test, y_pred))

pca_data_std = [np.array(value).std() for value in pca_data.values()]
pca_data_avg = [np.array(value).mean() for value in pca_data.values()]

fig, ax = plt.subplots()

ind = np.arange(6)    # the x locations for the groups
width = 0.35         # the width of the bars
ax.bar(ind, normal_data_avg, width, bottom=0,  yerr=normal_data_std ,label='Normal Data')


ax.bar(ind + width, pca_data_avg, width, bottom=0, yerr=pca_data_std, label='PCA Data')

ax.set_title('Accuracy by classifier with std ')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(list(normal_data.keys()))
ax.set_xlabel("Classifiers")
ax.set_ylabel("Accuracy")
ax.legend()

ax.autoscale_view()

plt.savefig("GeneralAccuracy_BarPlot_ErrorBar.pdf")
plt.close()

x = np.arange(N)
fig, ax = plt.subplots(nrows=2, ncols=1)
color=["red", "green", "blue", "orange", "yellow" ,"black"]
for i,clr in zip(normal_data.keys(),color):
    ax[0].errorbar(x, normal_data[i] ,  label=i, c=clr)

for i,clr in zip(pca_data.keys(),color):
    ax[1].errorbar(x, pca_data[i] ,  label=i, c=clr)

ax[0].set_title('Each iteration for classifier accuracy')

ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Accuracy")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Accuracy")
ax[0].legend()
ax[1].legend()

plt.savefig("Iter_by_iter.pdf")
plt.close()