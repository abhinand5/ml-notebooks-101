
# Introduction

[Kaggle](https://www.kaggle.com/) is an online community devoted to Data Science and Machine Learning founded by Google in 2010. It is the largest data community in the world with members ranging from ML beginners like yourself to some of the best researchers in the world. Kaggle is also the best place to start playing with data as it hosts over **23,000 public datasets** and more than **200,000 public notebooks**  that can be run online! And in case that’s not enough, Kaggle also hosts many Data Science competitions with insanely high cash prizes (1.5 Million was offered once!).

![Alt text](https://miro.medium.com/max/800/0*ftOal7fKVCNtJr4N.png)

Kaggle Competitions are a great way to test your knowledge and see where you stand in the Data Science world! If you are a beginner, you should start by practicing the old competition problems like [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic). After that, you can move on to the active competitions and maybe even win huge cash prices!!!

[**Titanic: Machine Learning from Disaster**](https://www.kaggle.com/c/titanic): This challenge is a very popular beginner project for ML as it has multiple tutorials available. So, it is a great introduction to ML concepts like data exploration, feature engineering, and model tuning.

# Challenge

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we have to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (i.e. name, age, gender, socio-economic class, etc).

![Alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg)

# Data Description
## Overview
The data has been split into two groups:
* training set (train.csv)
* test set (test.csv)

The training set should be used to build our machine learning models. For the training set, the outcome (also known as the “ground truth”) for each passenger is already provided. Our model will be based on “features” like passengers’ gender and class. We can also use [feature engineering](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/) to create new features.

The test set should be used to see how well our model performs on unseen data. For the test set, the ground truth for each passenger is not provided. It is our job to predict these outcomes. For each passenger in the test set, we use the model that we trained to predict whether or not they survived the sinking of the Titanic.

## Data Dictionary


| Variable | Definition |  Key |
| ------ | ------ | ------ |
| survival | Survival | 0 = No, 1 = Yes |
| pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |
| sex | Sex |  |
| Age | Age in years |  |
| sibsp | # of siblings / spouses aboard the Titanic |  |
| parch | # of parents / children aboard the Titanic |  |
| ticket | Ticket number |  |
| fare | Passenger fare |  |
| cabin | Cabin number |  |
| embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |

## Variable Notes
**pclass:** A proxy for socio-economic status (SES)  
1st = Upper  
2nd = Middle  
3rd = Lower  
  
**age:** Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5  
  
**sibsp:** The dataset defines family relations in this way...  
Sibling = brother, sister, stepbrother, stepsister  
Spouse = husband, wife (mistresses and fiancés were ignored)  
  
**parch:** The dataset defines family relations in this way...  
Parent = mother, father  
Child = daughter, son, stepdaughter, stepson  
Some children travelled only with a nanny, therefore parch=0 for them.

# Solution

## Importing the Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
## Importing the Data
```python
dataset = pd.read_csv("train.csv")
X = dataset.iloc[:,[2,4,5,6,7,9,11]].values
Y = dataset.iloc[:,1].values
testdataset = pd.read_csv("test.csv")
X_test = testdataset.iloc[:,[1,3,4,5,6,8,10]].values
```

## Data Preprocessing

### Missing Values
```python
from sklearn.impute import SimpleImputer
missingvalues1 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
missingvalues1 = missingvalues1.fit(X[:,[2]])
X[:,[2]] = missingvalues1.transform(X[:,[2]])
missingvalues2 = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
missingvalues2 = missingvalues2.fit(X[:,[6]])
X[:,[6]] = missingvalues2.transform(X[:,[6]])
missingvalues3 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
missingvalues3 = missingvalues3.fit(X_test[:,[2]])
X_test[:,[2]] = missingvalues3.transform(X_test[:,[2]])
missingvalues4 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
missingvalues4 = missingvalues4.fit(X_test[:,5]])
X_test[:,[5]] = missingvalues4.transform(X_test[:,[5]])
```
### Encoding Categorical Data
```python
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder1 = LabelEncoder()
X[:, 1] = labelencoder1.fit_transform(X[:,1])
labelencoder2 = LabelEncoder()
X[:, 6] = labelencoder2.fit_transform(X[:,6])
X = X.astype(float)
onehotencoder1 = OneHotEncoder(categorical_features = [1])
X = onehotencoder1.fit_transform(X).toarray()
X = X[:,1:]
onehotencoder2 = OneHotEncoder(categorical_features = [6])
X = onehotencoder2.fit_transform(X).toarray()
X = X[:,1:]
labelencoder3 = LabelEncoder()
X_test[:, 2] = labelencoder3.fit_transform (X_test[:,2])
labelencoder4 = LabelEncoder()
X_test[:, 6] = labelencoder4.fit_transform (X_test[:,6])
X_test = X_test.astype(float)
onehotencoder3 = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder3.fit_transform(X_test).toarray()
X_test = X_test[:,1:]
onehotencoder4 = OneHotEncoder(categorical_features = [6])
X_test = onehotencoder4.fit_transform(X_test).toarray()
X_test = X_test[:,1:]
```
### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)
```
## Building the Machine Learning Models (Classification)
### KNN
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
knn.fit(X, Y)
Y_pred = knn.predict(X_test)
```
### SVM
```python
from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=0)
svm.fit(X, Y)
Y_pred = svm.predict(X_test)
```
### Naïve Bayes
```python
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(X, Y)
Y_pred = naive.predict(X_test)
```
### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
decision = DecisionTreeClassifier(criterion='gini', random_state=0)
decision.fit(X, Y)
Y_pred = decision.predict(X_test)
```
### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rndom = RandomForestClassifier(n_estimators=10, criterion="gini", random_state=0)
rndom.fit(X, Y)
Y_pred = rndom.predict(X_test)
```
### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X, Y)
Y_pred = logistic.predict(X_test)
```
### ANN
```python
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X,Y,batch_size = 10,epochs = 100)
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
Y_pred = Y_pred.astype(int)
```
## Prediction

### Which Model is the best?

| Model | Accuracy |
| ------ | ------ |
| KNN | 75.598 |
| SVM | 76.555 |
| Naïve Bayes | 75.598 |
| Decision Tree | 71.291 |
| Random Forest | 74.162 |
| Logistic Regression | 76.076 |
| ANN| 75.598 |

As we can see SVM Classifier shows the best result. Therefore, **SVM Classifier Model** is the best model for this Titanic: Machine Learning from disaster Challenge.
