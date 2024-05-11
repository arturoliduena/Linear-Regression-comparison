from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eig, inv
import statsmodels.api as sm
import pandas as pd
import numpy as np

def compute_k1(p, var, alpha_OLS):
    k_1=p*var/(alpha_OLS.T*alpha_OLS)
    x = k_1.tolist()[0][0]
    print("k_1 estimator:\t\t{}".format(x))
    return x

def compute_k1_AD(var, alpha_OLS, lambda_max):
    k_1AD = 2/(8*lambda_max) * np.sum(var/alpha_OLS.A1**2 )
    print("k_1(AD) estimator:\t{}".format(np.real(k_1AD)))

def compute_k2_AD(var, alpha_OLS, lambda_max):
    k_2AD = np.median([2*var/(lambda_max*alpha_OLS.A1**2)])
    print("k_2(AD) estimator:\t{}".format(np.real(k_2AD)))

def compute_k3_AD(p, var, alpha_OLS, lambda_max):
    k_3AD = 2*var/(lambda_max*np.prod([x**2 for x in alpha_OLS])**1/p)
    print("k_3(AD) estimator:\t{}".format(np.real(k_3AD.tolist()[0][0])))

def compute_k4_AD(p, var, alpha_OLS, lambda_max):
    k_4AD = (2*p/lambda_max) * sum([var/x**2 for x in alpha_OLS])
    print("k_4(AD) estimator:\t{}".format(np.real(k_4AD.tolist()[0][0])))


if __name__ == "__main__":
    # Load dataset
    df = fetch_california_housing(as_frame=True).frame
    # Prepare data columns (i.e. eliminate target variable)
    data_columns = list(df.columns)[:-1]
    # Standardize the data to adequately compute the estimators
    df_standarized = df.copy()
    df_standarized[list(df.columns)] = StandardScaler(
    ).fit_transform(df[list(df.columns)])
    # Compute eigenvectors and eigenvalues for the covariance matrix associated to the dataset
    X = np.asmatrix(df_standarized[data_columns].values)
    A = X.T * X
    lambdas, T = eig(A)
    # Find the lambda_max parameter used to compute the estimators
    lambda_max = max(lambdas)
    # Properly set the dimensions of the matrixes for future operations
    T = T.T
    Z = X * T
    # Extract the target value and prepare for future operations
    Y = np.asmatrix(df_standarized.MedHouseVal.values).T
    # Compute the OLS of the data; it will be used as reference
    model = sm.OLS(df_standarized.MedHouseVal,df_standarized[data_columns])
    results = model.fit()
    # Get the betas corresponding to the OLS fit and set dimensions correctly
    beta_OLS = np.asmatrix(results.params).T
    # Compute lambdas associated to the betas from OLS fit
    alpha_OLS = (inv(T) * beta_OLS)
    n, p = df_standarized.shape[0], len(data_columns)
    # Compute variance
    var = (Y.T*Y - (alpha_OLS.T * Z.T) * Y)/(1)

    compute_k1(p, var, alpha_OLS)

    compute_k1_AD(var, alpha_OLS, lambda_max)
    
    compute_k2_AD(var, alpha_OLS, lambda_max)

    compute_k3_AD(p, var, alpha_OLS, lambda_max)
    
    compute_k4_AD(p, var, alpha_OLS, lambda_max)

