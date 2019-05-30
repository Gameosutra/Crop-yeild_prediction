# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('updated3.csv')
y=pd.DataFrame(dataset.iloc[:, 8].values)
X=pd.DataFrame(dataset.iloc[:, 0:8].values)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.values[:, 0] = labelencoder_X.fit_transform(X.values[:, 0])
X.values[:, 2] = labelencoder_X.fit_transform(X.values[:, 2])
X.values[:, 3] = labelencoder_X.fit_transform(X.values[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [0])
X= onehotencoder.fit_transform(X).toarray()

#avoiding the dumny variable trap
X= X[:, 1:]

#splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.2, random_state= 0)

#scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y=StandardScaler()
X_train= sc_x.fit_transform(X_train)
y_train= sc_y.fit_transform(y_train)
X_test= sc_x.fit_transform(X_test)
y_test= sc_y.fit_transform(y_test)

#fitting multiple lr to training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#predicting test set results
y_pred= regressor.predict(X_test)

#variable
from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)

#accuracy
from sklearn.metrics import accuracy_score
y_test=y_test.astype('int')
y_pred=y_pred.astype('int')
print('accuracy score:', (accuracy_score(y_test,y_pred)*100),'%')

#rms
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))

#median absolute error
from sklearn.metrics import median_absolute_error
median_absolute_error(y_test, y_pred)

#mean absolute error
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)