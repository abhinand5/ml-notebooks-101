# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training data
dataset = pd.read_csv("train.csv")
X = dataset.iloc[:,[2,4,5,6,7,9,11]].values
Y = dataset.iloc[:,1].values

# importing the testing data
testdataset = pd.read_csv("test.csv")
X_test = testdataset.iloc[:,[1,3,4,5,6,8,10]].values

# handling missing values in the dataset
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
missingvalues4 = missingvalues4.fit(X_test[:,[5]])
X_test[:,[5]] = missingvalues4.transform(X_test[:,[5]])

# encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder1 = LabelEncoder()
X[:, 1] = labelencoder1.fit_transform(X[:,1])
labelencoder2 = LabelEncoder()
X[:, 6] = labelencoder2.fit_transform(X[:,6])
X = X.astype(float)
onehotencoder1 = OneHotEncoder(categorical_features = [6])
X = onehotencoder1.fit_transform(X).toarray()
X = X[:,1:]
labelencoder3 = LabelEncoder()
X_test[:, 1] = labelencoder3.fit_transform (X_test[:,1])
labelencoder4 = LabelEncoder()
X_test[:, 6] = labelencoder4.fit_transform (X_test[:,6])
X_test = X_test.astype(float)
onehotencoder2 = OneHotEncoder(categorical_features = [6])
X_test = onehotencoder2.fit_transform(X_test).toarray()
X_test = X_test[:,1:]

# encoding categorical data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

# training a ML model
from sklearn.tree import DecisionTreeClassifier
decision = DecisionTreeClassifier(criterion='gini', random_state=0)
decision.fit(X, Y)

# predicting the results
Y_pred = decision.predict(X_test)