#svr

#import libraries
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

#splitting data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.2, random_state= 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y=StandardScaler()
X_train= sc_x.fit_transform(X_train)
y_train= sc_y.fit_transform(y_train)
X_test= sc_x.fit_transform(X_test)
y_test= sc_y.fit_transform(y_test)
#y_test= sc_y.inverse_transform(sc_y.fit_transform(y_test))          

#fitting svr model to dataset
from sklearn.svm import SVR
regressor= SVR(kernel= 'linear')
regressor.fit(X_train,y_train)

#prediction svr
#y_pred=sc_y.inverse_transform(regressor.predict(X_test))
y_pred=regressor.predict(X_test)


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

#accuracy2
from sklearn.model_selection import cross_val_score
from sklearn import svm
y_test=y_test.astype('int')
y_pred=y_pred.astype('int')
clf = svm.SVC(kernel='svm').fit(X_train, y_train)
clf.score(y_test,y_pred)

'''#Building and training a Logistic Regression model
import statsmodels.formula.api as sm
logistic1 = sm.logit(formula='Yield~Production/Area',data=dataset)
fitted1 = logistic1.fit()
fitted1.summary()

###predicting values
predicted_values1=fitted1.predict(dataset[["Area"]+['Production']+['Water Quality']+['Minimum Selling Price']+['Rainfall']])
predicted_values1[1:10]

#Confusion matrix, Accuracy, sensitivity and specificity
from sklearn.metrics import confusion_matrix
X.shape
y.shape
X_train.shape
y_test.shape
y_pred.shape
cm1 = confusion_matrix(dataset[['Yield']],y_pred)
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))
#####from confusion matrix calculate accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity1)'''