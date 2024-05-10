import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

def load_housing():
  #data
  california_housing = fetch_california_housing(as_frame=True)
  print(california_housing.DESCR)
  california_housing.frame.head()
  california_housing.data.head()
  california_housing.frame.info()
  features = california_housing.feature_names
  print(f"feature: {features}")
  #splot
  X_train, X_test, y_train, y_test = train_test_split(california_housing.data, california_housing.target, test_size=0.3, random_state=17)
  print("The dimension of X_train is {}".format(X_train.shape))
  print("The dimension of X_test is {}".format(X_test.shape))
  print("The dimension of y_train is {}".format(y_train.shape))
  print("The dimension of y_test is {}".format(y_test.shape))
  
  return features, X_train, X_test, y_train, y_test

if __name__ == "__main__":
  load_housing()
  print("Dataset loaded successfully!")


