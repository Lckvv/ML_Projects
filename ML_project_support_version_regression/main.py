# SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:3].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y)

#Fitting svr to the dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X,Y)

#predict a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualising Regression the SVR results
plt.scatter(X,Y, color = 'red')
plt.plot(X,regressor.predict(X))
plt.title('SVR result')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

