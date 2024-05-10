from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dataset import load_housing

def linear_regression(features, X_train, X_test, y_train, y_test):
  #Model
  lr = LinearRegression()

  #Fit model
  lr.fit(X_train, y_train)

  #predict
  #prediction = lr.predict(X_test)

  train_score_lr = lr.score(X_train, y_train)
  test_score_lr = lr.score(X_test, y_test)

  print("The train score for lr model is {}".format(train_score_lr))
  print("The test score for lr model is {}".format(test_score_lr))


if __name__ == "__main__":
  features, X_train, X_test, y_train, y_test = load_housing()
  linear_regression(features, X_train, X_test, y_train, y_test)
  print("Linear Regression Model!")