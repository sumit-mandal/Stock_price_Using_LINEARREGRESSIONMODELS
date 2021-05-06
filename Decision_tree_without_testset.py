import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset = pd.read_csv('NFLX.csv')
X = dataset.iloc[:,2:3].values
y = dataset.iloc[:,3].values

#Fitting the decision tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

y_pred = regressor.predict([[300]])

#Visualising the graph
X_grid = np.arange(min(X),max(X),0.01) #Return evenly spaced values within a given interva
X_grid = X_grid.reshape((len(X_grid),1))
# we need to minimise the step size so we use minimum range and maximum range of X with stepsize of 0.1
#we reshape X_grid into a matrix where number of lines is the number of element of x_grid i.e. len(X_grid) and number of column which is 1
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')

plt.title('Stock prediction(Decison tree)')

plt.xlabel('High')
plt.ylabel('Low')
plt.show()
