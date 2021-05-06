# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('NFLX.csv')
X = dataset.iloc[:,3:4].values #independent vriables
y = dataset.iloc[:,5].values #dependent variables

#on looking output we find X is matrix as we have 1 column whereas in y we have no column therefore it is vector

#splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test =  train_test_split(X, y, test_size = 1/3, random_state = 0)

#X_train is matrix of independent variable and y_train is dependent variable factor for trainig set
#X_test is matrix of independent variable and y_test is dependent variable factor for test set


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#fitting Simple Linear Regression into the training set

from sklearn.linear_model import LinearRegression#odinary least ssquares linear regression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test results
y_pred = regressor.predict(X_test) #y_pred is vector of predictions of dependent variable
 #it will contain predicted salary of all the test set. since dependent variable is salary 
 
 #ypred contains predicted sal and y_test actual salary
 
 #predictionn of training set
y_pred_train = regressor.predict(X_train)
 
 #visualising the training set results

plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,y_pred_train,color = 'blue')
plt.title('Stock prediction(TRAINING_SET)')

plt.xlabel('Low')
plt.ylabel('close')
plt.show()


#Visualising test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Stock prediction(Test_SET)')

plt.xlabel('Low')
plt.ylabel('close')
plt.show()



