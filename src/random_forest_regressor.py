# Tratamiento de datos
# ==============================================================================
import warnings
import optuna
from joblib import Parallel, delayed, cpu_count
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 8


# ==============================================================================
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"Versión de scikit-learn: {sklearn.__version__}")


def run_random_forest(X_train, y_train, X_test, y_test, n_estimators, max_depth):
    # ==============================================================================
    modelo = RandomForestRegressor(
        n_estimators=10,
        criterion='squared_error',
        max_depth=None,
        max_features=1,
        oob_score=False,
        n_jobs=-1,
        random_state=123
    )

    # ==============================================================================
    modelo.fit(X_train, y_train)

    # ==============================================================================
    predicciones = modelo.predict(X=X_test)
    rmse = root_mean_squared_error(y_true=y_test, y_pred=predicciones)
    print(f"El error (rmse) de test es: {rmse}")


if __name__ == "__main__":
    # data
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    # ==============================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=123
    )
    print(f"Tamaño conjunto entrenamiento: {X_train.shape[0]}")
    print(f"Tamaño conjunto test: {X_test.shape[0]}")
    run_random_forest(X_train, y_train, X_test, y_test, 10, None)
