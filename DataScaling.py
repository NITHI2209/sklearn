#Data scaling is divided into 2 types -> standardscaler and minmax scaler
import pandas as pd
from sklearn.datasets import fetch_openml
df = fetch_openml("titanic",version= 1 , as_frame=True)["data"]
from sklearn.preprocessing import StandardScaler
num_col = df.select_dtypes(include=["int64","float64","int32"]).columns
ss = StandardScaler()
df[num_col] = ss.fit_transform(df[num_col])
print(df[num_col].describe())

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
df[num_col] = minmax.fit_transform(df[num_col])
print(df[num_col])