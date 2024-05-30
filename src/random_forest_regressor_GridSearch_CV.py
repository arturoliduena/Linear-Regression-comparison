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
    param_grid = {'n_estimators': [150],
                  'max_features': [5, 7, 9],
                  'max_depth': [None, 3, 10, 20]
                  }

    # Búsqueda por grid search con validación cruzada
    # ==============================================================================
    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=123),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        n_jobs=cpu_count() - 1,
        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
        refit=True,
        verbose=0,
        return_train_score=True
    )

    grid.fit(X=X_train, y=y_train)

    # Resultados
    # ==============================================================================
    resultados = pd.DataFrame(grid.cv_results_)
    tabular = resultados.filter(regex='(param.*|mean_t|std_t)') \
        .drop(columns='params') \
        .sort_values('mean_test_score', ascending=False) \
        .head(4)
    print(tabular)

    # Mejores hiperparámetros encontrados mediante validación cruzada
    # ==============================================================================
    print("----------------------------------------")
    print("Mejores hiperparámetros encontrados (cv)")
    print("----------------------------------------")
    print(grid.best_params_, ":", grid.best_score_, grid.scoring)

    # Error de test del modelo final
    # ==============================================================================
    modelo_final = grid.best_estimator_
    predicciones = modelo_final.predict(X=X_test)
    rmse = root_mean_squared_error(
        y_true=y_test,
        y_pred=predicciones,
    )
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
    run_random_forest(X_train, y_train, X_test, y_test)
