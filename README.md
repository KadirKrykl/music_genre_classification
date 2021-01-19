# music_genre_classification

# Description

To manually classify a music sample or song, one needs to listen to the song and select the genre. This process takes a lot of time one should also know about different types of music. Being able to instantly classify songs in any given playlist or library by genre is an important functionality for any music streaming or purchasing service. Our system takes the piece of music, then converts it into numerical values with some operations, then determines the possible genre by comparing it with our data. Python has a library called "Librosa" to convert audio files to numeric data, from which we can create the numerical data we need.

# Team & Roles

Kadir Karayakalı 160709052 -> Data Preprocessing - SVM 

Berkcan Erguncu 160709005 -> PCA - Ensemble Learning-based Classification

Mandana Zooyousefin 180709711 -> KNN Classification - Neural Network-based Classification


# Structure

Scripts written for preprocess, pca and classification steps are in the SectionalFilesForReport file. Classifier file, on the other hand, is an easy-to-run state that contains all the operations. In order to run the system, processing is required in test.py file.



# Language, version, and main file

We used Python language and its version 3.8.5. Our main file is test.py.

The file to be tested should be written in the "song.wav" section in the code below.
```
filePath = os.path.join(workingPath, "song.wav")
```

Then one of the KNN, SVM and MLP methods should be written in the classfier section. If you want to apply bagging, you should write True in the bagging section of the line, if not False.
```
classifier = Classifier(classifier = "method", bagging = True/False)
```



