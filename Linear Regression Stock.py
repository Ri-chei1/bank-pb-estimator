import pandas as pd
import numpy as np


df = pd.read_csv("bank_pbs.csv")

## Take only the necessary Data
d = df[["Ticker", "ROE", "Log_Assets", "Log_pb", "eq_ratio"]].copy()

## Splitting the data to X and y values
X = d[["ROE", "Log_Assets", "eq_ratio"]].copy()
y = d["Log_pb"].copy()


## Splitting Train and Test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Linear Regression
class LinearRegressionClosed:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        beta = np.linalg.inv(Xb.T @ Xb) @ (Xb.T) @ y
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

    def predict(self, X):
        X = np.array(X)

        return X @ self.coef_ + self.intercept_


reg = LinearRegressionClosed()

reg.fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)


data = {
    "Training MSE": [train_mse],
    "Training R2": [train_r2],
    "Test MSE": [test_mse],
    "Test R2": [test_r2]
}
results_df = pd.DataFrame(data, index = ['Linear Regression'])

print (results_df)

