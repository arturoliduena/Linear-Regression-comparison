import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from dataset import load_housing

def lasso_linear_regression(features, X_train, X_test, y_train, y_test):
  #Lasso regression model
  lasso = Lasso(alpha = 10)
  lasso.fit(X_train,y_train)
  train_score_ls =lasso.score(X_train,y_train)
  test_score_ls =lasso.score(X_test,y_test)

  print("The train score for ls model is {}".format(train_score_ls))
  print("The test score for ls model is {}".format(test_score_ls))

  pd.Series(lasso.coef_, features).sort_values(ascending = True).plot(kind = "bar")

  #Lasso Cross validation
  lasso_cv = LassoCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10], random_state=0).fit(X_train, y_train)

  #score
  print("The train score for lasso_cv model is {}".format(lasso_cv.score(X_train, y_train)))
  print("The train score for lasso_cv model is {}".format(lasso_cv.score(X_test, y_test)))


if __name__ == "__main__":
  features, X_train, X_test, y_train, y_test = load_housing()
  lasso_linear_regression(features, X_train, X_test, y_train, y_test)
  print("Model run successfully!")