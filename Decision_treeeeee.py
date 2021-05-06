# importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('NFLX.csv')
X = dataset.iloc[:,2:3].values
y = dataset.iloc[:,3].values

#splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test =  train_test_split(X, y, test_size = 1/3, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#Fitting the decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)


 #predictionn of training set
y_pred_train = regressor.predict(X_train)


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

