from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import fetch_california_housing

def models():
  california_housing = fetch_california_housing()
  print(california_housing.DESCR)
  # Define a dictionary of estimators with their hyperparameters
  estimators = {
    'LinearRegression': (LinearRegression(), {}),
    'Lasso': (Lasso(alpha=0.1), {'alpha': 0.1}),
    'Ridge': (Ridge(alpha=1.0), {'alpha': 1.0})
  }

  # Define evaluation metrics
  scoring_metrics = ['r2', 'neg_mean_squared_error', 'explained_variance']

  # Iterate over each estimator
  for estimator_name, (estimator_object, hyperparameters) in estimators.items():
    print(f"\n{estimator_name} Hyperparameters: {hyperparameters}")
    
    # Iterate over each scoring metric
    for metric in scoring_metrics:
      kfold = KFold(n_splits=10, random_state=11, shuffle=True)
      
      # Perform cross-validation
      scores = cross_val_score(estimator=estimator_object, X=california_housing.data, y=california_housing.target, cv=kfold, scoring=metric)
      
      # Print the evaluation metric
      print(f'{metric:>25}: mean score={scores.mean():.3f}, std={scores.std():.3f}')


if __name__ == "__main__":
  models()