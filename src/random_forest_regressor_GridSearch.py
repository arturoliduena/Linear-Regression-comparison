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


def run_random_forest(X_train, y_train, X_test, y_test):
    # Grid de hiperparámetros evaluados
    # ==============================================================================
    param_grid = ParameterGrid(
        {'n_estimators': [150],
         'max_features': [5, 7, 9],
         'max_depth': [None, 3, 10, 20]
         }
    )

    # Loop paralelizado para ajustar un modelo con cada combinación de hiperparámetros
    # ==============================================================================
    def eval_oob_error(X, y, params, verbose=True):
        """
        Función para entrenar un modelo utilizando unos parámetros determinados
        y que devuelve el out-of-bag error
        """
        modelo = RandomForestRegressor(
            oob_score=True,
            n_jobs=-1,
            random_state=123,
            ** params
        )

        modelo.fit(X, y)

        if verbose:
            print(f"model: {params} ✓")

        return {'params': params, 'oob_r2': modelo.oob_score_}

    resultados = Parallel(n_jobs=cpu_count()-1)(
        delayed(eval_oob_error)(X_train, y_train, params)
        for params in param_grid
    )

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(resultados)
    resultados = pd.concat(
        [resultados, resultados['params'].apply(pd.Series)], axis=1)
    resultados = resultados.drop(columns='params')
    resultados = resultados.sort_values('oob_r2', ascending=False)
    print(resultados.head(4))
    # ==============================================================================
    print("--------------------------------------------")
    print("Mejores hiperparámetros encontrados (oob-r2)")
    print("--------------------------------------------")
    print(resultados.iloc[0, 0:])


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
    run_random_forest(X_train, y_train, X_test, y_test)
