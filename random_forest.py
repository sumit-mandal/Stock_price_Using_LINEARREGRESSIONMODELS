# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("NFLX.csv")
X = dataset.iloc[:,2:3].values
y = dataset.iloc[:,3].values

#Fitting the decision tree regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 31,criterion='mse',random_state=0) #estimator is number of trees in forest,
regressor.fit(X,y)

#predicting the result
y_pred = regressor.predict([[300]])



#Visualising the model
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
# we need to minimise the step size so we use minimum range and maximum range of X with stepsize of 0.1
#we reshape X_grid into a matrix where number of lines is the number of element of x_grid i.e. len(X_grid) and number of column which is 1
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')

plt.title('Stock prediction(Random Forest)')

plt.xlabel('High')
plt.ylabel('Low')
plt.show()


