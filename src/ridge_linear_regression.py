from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from dataset import load_housing
from sklearn.linear_model import RidgeCV

def ridge_linear_regression(features, X_train, X_test, y_train, y_test):  
  #Ridge Regression Model
  ridgeReg = Ridge(alpha=10)

  ridgeReg.fit(X_train,y_train)

  #train and test scorefor ridge regression
  train_score_ridge = ridgeReg.score(X_train, y_train)
  test_score_ridge = ridgeReg.score(X_test, y_test)

  print("\nRidge Model............................................\n")
  print("The train score for ridge model is {}".format(train_score_ridge))
  print("The test score for ridge model is {}".format(test_score_ridge))
  #Using the linear CV model

  #Lasso Cross validation
  ridge_cv = RidgeCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 10]).fit(X_train, y_train)

  #score
  print("The train score for ridge_cv model is {}".format(ridge_cv.score(X_train, y_train)))
  print("The train score for ridge_cv model is {}".format(ridge_cv.score(X_test, y_test)))

if __name__ == "__main__":
  features, X_train, X_test, y_train, y_test = load_housing()
  ridge_linear_regression(features, X_train, X_test, y_train, y_test)
  print("ridge Linear Regression Model!")