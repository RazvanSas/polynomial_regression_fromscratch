
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class LinearRegressionClosed:

    def __init__(self):
        self.__coef_ = None
        self.__intercept_ = 0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        Xb = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

        self.__intercept_ = A[0]
        self.__coef_ = A[1:]

    def predict(self, X):
        X = np.array(X)

        return X @ self.__coef_ + self.__intercept_

    def get_coef(self):
        return self.__coef_

    def get_intercept(self):
        return self.__intercept_

def polynomial_features(X, degree):
    X = np.array(X)
    t = X.copy()

    for i in range(2,degree+1):
        X = np.concatenate((X, t ** i), axis = 1)

    return X

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

X_poly = polynomial_features(X, degree = 4)

print(X_poly)

regressor = LinearRegressionClosed()
regressor.fit(X_poly, y)


plt.scatter(X, y, color = 'black')
plt.plot(X, regressor.predict(X_poly), color = 'red')
plt.show()







