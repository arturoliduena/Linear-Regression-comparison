from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso


def models():
    california_housing = fetch_california_housing()
    alphas = [("LassoCV", 0.0015953096839947316), ("LassoLarsCV", 0.0012314644830673947), ("k1", 1.7229933519274163),
              ("k1(AD)", 0.0010845229813710349), ("k2(AD)", 0.00016990296535820582), ("k3(AD)", 0.0001779808406195281), ("k4(AD)", 0.06940947080774623)]
    # Define a dictionary of estimators with their hyperparameters
    estimators = {
        'LinearRegression': (LinearRegression, [(None, None)]),
        'Lasso': (Lasso, alphas),
        'Ridge': (Ridge, alphas)
    }

    # Define evaluation metrics
    scoring_metrics = ['r2', 'neg_mean_squared_error', 'explained_variance']

    # Iterate over each estimator
    for estimator_name, (estimator_object, hyperparameters) in estimators.items():

        for (name, alpha) in hyperparameters:
            x = ("" if alpha is None else name)
            print(f"\n{estimator_name} Hyperparameters: {x}")

            # Iterate over each scoring metric
            for metric in scoring_metrics:
                kfold = KFold(n_splits=10, random_state=11, shuffle=True)

                # Perform cross-validation
                scores = cross_val_score(estimator=(estimator_object(alpha=alpha) if alpha != None else estimator_object()), X=california_housing.data,
                                         y=california_housing.target, cv=kfold, scoring=metric)

                # Print the evaluation metric
                print(
                    f'{metric:>25}: mean score={scores.mean():.3f}, std={scores.std():.3f}')


if __name__ == "__main__":
    models()
