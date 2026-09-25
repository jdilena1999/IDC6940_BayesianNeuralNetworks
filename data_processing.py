import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


data = pd.read_csv("radioml_features.csv")
y_col = ['SNR']
x_cols = [x for x in data.columns if x not in y_col]
y = data[y_col]
X = data[x_cols]


# X Features scaled down to mean = 0 and standard_deviation = 1
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)



