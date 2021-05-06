import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('NFLX.csv')
X = dataset.iloc[:,3:4].values #independent vriables
y = dataset.iloc[:,5].values #dependent variables

y = y.reshape(len(y),1)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#Fitting svr into dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[300]])))


#Visualising the svr result

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Stock prediction(TRAINING_SET)')

plt.xlabel('Low')
plt.ylabel('close')
plt.show()



