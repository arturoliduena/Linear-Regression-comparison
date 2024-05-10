import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import time

# Selecting Lasso via an information criterion
def lasso_linear_regression_criterion(X, y):
  start_time = time.time()
  lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion="aic")).fit(X, y)
  fit_time = time.time() - start_time

  # We store the AIC metric for each value of alpha used during fit
  results = pd.DataFrame(
    {
      "alphas": lasso_lars_ic[-1].alphas_,
      "AIC criterion": lasso_lars_ic[-1].criterion_,
    }
  )
  alpha_aic = lasso_lars_ic[-1].alpha_

  # Now, we perform the same analysis using the BIC criterion.
  lasso_lars_ic.set_params(lassolarsic__criterion="bic").fit(X, y)
  results["BIC criterion"] = lasso_lars_ic[-1].criterion_
  alpha_bic = lasso_lars_ic[-1].alpha_

  # Round the values in the DataFrame to two decimal points
  rounded_results = results.copy().round(4)
  # Create a figure and axis
  fig, ax = plt.subplots(figsize=(4, 6))
  ax.axis('off')
  # Create the table plot
  table = ax.table(cellText=rounded_results.values, colLabels=rounded_results.columns, loc='center')
  # Save the plot as an image
  plt.savefig('lasso_alpha.png')

  results = results.set_index("alphas")
  _ax = results.plot()
  _ax.vlines(
      alpha_aic,
      results["AIC criterion"].min(),
      results["AIC criterion"].max(),
      label="alpha: AIC estimate",
      linestyles="--",
      color="tab:blue",
  )
  _ax.vlines(
      alpha_bic,
      results["BIC criterion"].min(),
      results["BIC criterion"].max(),
      label="alpha: BIC estimate",
      linestyle="--",
      color="tab:orange",
  )
  _ax.set_xlabel(r"$\alpha$")
  _ax.set_ylabel("criterion")
  _ax.set_xscale("log")
  _ax.legend()
  _ = _ax.set_title(
      f"Information-criterion for model selection (training time {fit_time:.2f}s)"
  )

  # Show the plot
  plt.show()

# Selecting Lasso via cross-validation (Lasso via coordinate descent)
def lasso_linear_regression_CV(X, y):
  start_time = time.time()
  model = make_pipeline(StandardScaler(), LassoCV(cv=20)).fit(X, y)
  fit_time = time.time() - start_time

  lasso = model[-1]
  plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle=":")
  plt.plot(
      lasso.alphas_,
      lasso.mse_path_.mean(axis=-1),
      color="black",
      label="Average across the folds",
      linewidth=2,
  )
  plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha: CV estimate")

  plt.xlabel(r"$\alpha$")
  plt.ylabel("Mean square error")
  plt.legend()
  _ = plt.title(
      f"Mean square error on each fold: coordinate descent (train time: {fit_time:.2f}s)"
  )

  # Show the plot
  plt.show()

# Selecting Lasso via cross-validation (Lasso via least angle regression)
def lasso_linear_regression_LarsCV(X, y):
  start_time = time.time()
  model = make_pipeline(StandardScaler(), LassoLarsCV(cv=20)).fit(X, y)
  fit_time = time.time() - start_time

  lasso = model[-1]
  plt.semilogx(lasso.cv_alphas_, lasso.mse_path_, ":")
  plt.semilogx(
      lasso.cv_alphas_,
      lasso.mse_path_.mean(axis=-1),
      color="black",
      label="Average across the folds",
      linewidth=2,
  )
  plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha CV")

  plt.xlabel(r"$\alpha$")
  plt.ylabel("Mean square error")
  plt.legend()
  _ = plt.title(f"Mean square error on each fold: Lars (train time: {fit_time:.2f}s)")
  
  # Show the plot
  plt.show()

if __name__ == "__main__":
  #data
  X, y = fetch_california_housing(return_X_y=True, as_frame=True)
  rng = np.random.RandomState(42)
  n_random_features = 20
  X_random = pd.DataFrame(
      rng.randn(X.shape[0], n_random_features),
      columns=[f"random_{i:02d}" for i in range(n_random_features)],
  )
  X = pd.concat([X, X_random], axis=1)
  lasso_linear_regression_criterion(X, y)
  lasso_linear_regression_CV(X, y)
  lasso_linear_regression_LarsCV(X, y)