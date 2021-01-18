import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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
    'KNN': [],
    'KNN_PCA': [],
    'SVM': [],
    'SVM_PCA': [],
    'MLP': [],
    'MLP_PCA': [],
}
#For normal data
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf =  BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_estimators=4).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dict_for_df['KNN'].append(accuracy_score(y_test, y_pred))

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = BaggingClassifier(SVC(gamma='auto'), n_estimators=4).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dict_for_df['SVM'].append(accuracy_score(y_test, y_pred))

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = BaggingClassifier(MLPClassifier(hidden_layer_sizes=(256,128,64,8), max_iter=1000), n_estimators=4).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dict_for_df['MLP'].append(accuracy_score(y_test, y_pred))

#For data which applyed pca
pca = PCA(n_components = 15)
pca.fit(X)
reduced_X = pca.transform(X)
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(reduced_X, y, test_size=0.3)
    clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), n_estimators=4).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dict_for_df['KNN_PCA'].append(accuracy_score(y_test, y_pred))

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(reduced_X, y, test_size=0.3)
    clf = BaggingClassifier(SVC(gamma='auto'), n_estimators=4).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dict_for_df['SVM_PCA'].append(accuracy_score(y_test, y_pred))

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(reduced_X, y, test_size=0.3)
    clf = BaggingClassifier(MLPClassifier(hidden_layer_sizes=(256,128,64,8), max_iter=1000), n_estimators=4).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dict_for_df['MLP_PCA'].append(accuracy_score(y_test, y_pred))



df = pd.DataFrame.from_dict(dict_for_df)
graph = df.boxplot()
graph.set_ylabel('Accuracy (%)')
plt.show()
plt.savefig("BaggingClassifier.pdf")
plt.close()