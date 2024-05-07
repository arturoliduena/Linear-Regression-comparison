#libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes

#data
diabetes = load_diabetes()
diabetes_df=pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
#target variable
#represents the quantitative measure of disease progression one year after baseline for diabetes patients
diabetes_df['Disease']=diabetes.target

# Preview
print(diabetes_df.head())
print("description of the dataset")
print(diabetes_df.describe())
print("shape of the dataset")
print(diabetes_df.shape)
print("columns of the dataset")
print(diabetes_df.columns)
print("info of the dataset")
print(diabetes_df.info())
print("null values in the dataset")
print(diabetes_df.isnull().sum())

#Exploration
plt.figure(figsize = (10, 10))
sns.heatmap(diabetes_df.corr(), annot = True)
plt.title('Correlation Heatmap of Diabetes Dataset')
plt.show()

