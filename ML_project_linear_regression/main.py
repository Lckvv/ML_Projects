#LinearRegression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Splitting the dataset info the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=np.random)

# #Feature Scalling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,Y_train)

#Predicting the Test set result
y_pred = regression.predict(X_test)

#Visualising the Trening set result
plt.scatter(X_train,Y_train, edgecolors='black')
plt.plot(X_train, regression.predict(X_train), 'green')
plt.show()

#Visualising the Test set result
plt.scatter(X_test,Y_test, edgecolors='black')
plt.plot(X_train, regression.predict(X_train), 'green')
plt.show()