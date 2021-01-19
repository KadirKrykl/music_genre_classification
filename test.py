from Classifier import Classifier
import os

workingPath = os.getcwd()
filePath = os.path.join(workingPath, "blues.wav")

#Examples 
classifier = Classifier(classifier = "KNN", bagging = True)
classifier.train()
print(classifier.predict(filePath))

classifier = Classifier(classifier = "SVM", bagging = True)
classifier.train()
print(classifier.predict(filePath))

classifier = Classifier(classifier = "MLP", bagging = False)
classifier.train()
print(classifier.predict(filePath))