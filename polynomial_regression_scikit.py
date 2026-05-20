import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

reg_poly = PolynomialFeatures(degree = 4)
X_poly = reg_poly.fit_transform(X)

linear_regressor = LinearRegression()
polynomial_regressor = LinearRegression()

linear_regressor.fit(X, y)
polynomial_regressor.fit(X_poly, y)

plt.scatter(X, y, color = 'black')
plt.plot(X, linear_regressor.predict(X), color = 'red')
plt.show()


X = X.reshape(len(X),)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'black')
plt.plot(X_grid, polynomial_regressor.predict(reg_poly.fit_transform(X_grid)), color = 'red')
plt.show()




