#importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset's
train = pd.read_csv("/home/virutyagi/Projects/KaggleProjects/Titanic Problem/train.csv")
test = pd.read_csv("/home/virutyagi/Projects/KaggleProjects/Titanic Problem/test.csv")
train.head(5)
test.head(5)

#diceing and sliceing

X = train.iloc[:,[2,4,5,6,7,9,11]].values
y = train.iloc[:,1].values
X_test = test.iloc[:,[1,3,4,5,6,8,10]].values

#visualization 
plt.scatter(X[y==0,2],X[y==0,0],color = 'r')
plt.scatter(X[y==1,2],X[y==1,0],color = 'g')
plt.show()

#handling missing values

from sklearn.preprocessing import Imputer
imp = Imputer()
tesst = X[:,2].reshape(-1,1)
tesst = imp.fit_transform(tesst)
X[:,2] = tesst.ravel()
X_test[:,[2,5]] = imp.fit_transform(X_test[:,[2,5]])
del(tesst)

train1 = pd.DataFrame(X[:,[1,6]])
train1.describe()
test1 = pd.DataFrame(X_test[:,[1,6]])
test1.describe()
train1[1] = train1[1].fillna("S")
X[:,[1,6]] = train1

#converting categorical values into 
from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[:,1] = lab.fit_transform(X[:,1])
X[:,6] = lab.fit_transform(X[:,6])

X_test[:,1] = lab.fit_transform(X_test[:,1])
X_test[:,6] = lab.fit_transform(X_test[:,6])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,6])
X = one.fit_transform(X)
X_test = one.fit_transform(X_test)
X = X.toarray()
X_test = X_test.toarray()

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X,y)
DT.score(X,y)
y_test = DT.predict(X_test)
DT.score(X_test,y_test)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,y)
log_reg.score(X,y)
y_test1 = log_reg.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X,y)
y_test1 = knn.predict(X_test)


Predictedvalue  = pd.DataFrame(y_test1)

train.describe()
#storing predicted values into the csv file
test['Survived'] = Predictedvalue
test.to_csv('/home/virutyagi/Desktop/Submission.csv')
