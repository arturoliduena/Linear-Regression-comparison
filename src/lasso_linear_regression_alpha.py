import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import time

# Selecting Lasso via cross-validation (Lasso via coordinate descent)
def lasso_linear_regression_CV(X, y):
    start_time = time.time()
    model = make_pipeline(StandardScaler(), LassoCV(
        tol=1e-9, max_iter=5000, cv=20, n_jobs=-1)).fit(X, y)
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
    print(lasso.alpha_)
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha CV")

    plt.xlabel(r"$\alpha$")
    plt.ylabel("MSE")
    plt.legend()
    _ = plt.title(
        f"MSE on each fold: Coordinate Descent (train time: {fit_time:.2f}s)"
    )

    # Show the plot
    plt.show()

# Selecting Lasso via cross-validation (Lasso via least angle regression)
def lasso_linear_regression_LarsCV(X, y):
    start_time = time.time()
    model = make_pipeline(StandardScaler(), LassoLarsCV(
        max_iter=5000, cv=20, n_jobs=-1)).fit(X, y)
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
    print(lasso.alpha_)
    plt.axvline(lasso.alpha_, linestyle="--", color="black", label="alpha CV")

    plt.xlabel(r"$\alpha$")
    plt.ylabel("MSE")
    plt.legend()
    _ = plt.title(f"MSE on each fold: LARS (train time: {fit_time:.2f}s)")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # data
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    rng = np.random.RandomState(42)
    # n_random_features = 20
    # X_random = pd.DataFrame(
    #     rng.randn(X.shape[0], n_random_features),
    #     columns=[f"random_{i:02d}" for i in range(n_random_features)],
    # )
    # X = pd.concat([X, X_random], axis=1)
    lasso_linear_regression_CV(X, y)
    lasso_linear_regression_LarsCV(X, y)
