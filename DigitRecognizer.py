import numpy as numpy
import pandas as pd 
import matplotlib.pyplot as plt 

train = pd.read_csv("/home/virutyagi/Projects/KaggleProjects/DigitRecognizer/train.csv")
test = pd.read_csv("/home/virutyagi/Projects/KaggleProjects/DigitRecognizer/test.csv")

y = train.iloc[:,0].values
X_train = train.iloc[:,1:].values
X_test = test.iloc[:,:].values

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y)
knn.score(X_train,y)

y_test = knn.predict(X_test)