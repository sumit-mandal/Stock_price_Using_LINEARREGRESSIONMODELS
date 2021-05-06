# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('NFLX.csv')
X = dataset.iloc[:,2:3].values #independent vriables
y = dataset.iloc[:,4].values

#training the Linear Regressionn model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Training the polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualising the Linear Regression results

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Stock prediction(Linear Regression)')
plt.xlabel('Low')
plt.ylabel('close')
plt.show()

#visualising the polynomial_regression result
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Stock prediction(Polynomial Regression)')
plt.xlabel('Low')
plt.ylabel('close')
plt.show()




# Predicting a new result with Linear Regression
lin_reg.predict([[307]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[307]]))

#Visualising the graph in higher resolution
X_grid = np.arange(min(X),max(X),0.1)
# we need to minimise the step size so we use minimum range and maximum range of X with stepsize of 0.1
X_grid = X_grid.reshape((len(X_grid),1))
#we reshape X_grid into a matrix where number of lines is the number of element of x_grid i.e. len(X_grid) and number of column which is 1
plt.scatter(X,y,color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')

plt.title('Stock prediction(TRAINING_SET)')

plt.xlabel('Low')
plt.ylabel('Close')
plt.show()

