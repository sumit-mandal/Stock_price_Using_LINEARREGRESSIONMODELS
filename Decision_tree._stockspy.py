# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('FB_stock.csv')
X = dataset.iloc[:,3:4].values #independent vriables
y = dataset.iloc[:,5].values #dependent variable

#Fitting the decision tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X,y)

#Predicting a new result
y_pred = regressor.predict([[6.5]])

#Visualising the decision tree regression reults
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or bluff(Regresion model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
