from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dataset import load_housing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def linear_regression(features, X_train, X_test, y_train, y_test):
  #Model
  regr = LinearRegression()

  # Train the model using the training sets
  regr.fit(X_train, y_train)

  # Make predictions using the testing set
  y_pred = regr.predict(X_test)

  print("The train score for regr model is {}".format(regr.score(X_train, y_train)))
  print("The test score for regr model is {}".format(regr.score(X_test, y_test)))

  # The coefficients
  print("Coefficients: \n", regr.coef_)
  # The mean squared error
  print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
  # The coefficient of determination: 1 is perfect prediction
  print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

  # Plot outputs
  # plt.scatter(X_test, y_test, color="black")
  # plt.plot(X_test, y_pred, color="blue", linewidth=3)

  # plt.xticks(())
  # plt.yticks(())

  # plt.show()

if __name__ == "__main__":
  features, X_train, X_test, y_train, y_test = load_housing()
  linear_regression(features, X_train, X_test, y_train, y_test)
  print("Linear Regression Model!")