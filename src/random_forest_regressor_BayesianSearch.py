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
    # Búsqueda bayesiana de hiperparámetros con optuna
    # ==============================================================================
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
            'max_features': trial.suggest_float('max_features', 0.2, 1.0),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 1),
            # Fixed parameters
            'n_jobs': -1,
            'random_state': 4576688,

        }

        modelo = RandomForestRegressor(**params)
        cross_val_scores = cross_val_score(
            estimator=modelo,
            X=X_train,
            y=y_train,
            cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        score = np.mean(cross_val_scores)
        return score

    # Se maximiza por que el score es negativo
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30,
                   show_progress_bar=True, timeout=60*5)

    print('Mejores hyperparameters:', study.best_params)
    print('Mejor score:', study.best_value)

    # Error de test del modelo final
    # ==============================================================================
    modelo_final = RandomForestRegressor(**study.best_params)
    modelo_final.fit(X_train, y_train)
    predicciones = modelo_final.predict(X=X_test)
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
    run_random_forest(X_train, y_train, X_test, y_test)
