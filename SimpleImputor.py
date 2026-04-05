import pandas as pd
from sklearn.datasets import fetch_openml
df = fetch_openml("titanic",version= 1 , as_frame=True)["data"]
from sklearn.impute import SimpleImputer
print(f'Number of null values before imputing:{df.age.isnull().sum()}')
imp = SimpleImputer(strategy="mean")
df["age"] = imp.fit_transform(df[["age"]])
print(f'Number of null values after imputing:{df.age.isnull().sum()}')
impo = SimpleImputer(strategy="most_frequent")
df["cabin"] = impo.fit_transform(df[["cabin"]]).ravel()
print(f'Number of null values after imputing:{df.cabin.isnull().sum()}')
