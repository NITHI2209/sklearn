import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
df = fetch_openml("titanic",version= 1 , as_frame=True)["data"]
from sklearn.preprocessing import OneHotEncoder
df[["female","male"]] = OneHotEncoder().fit_transform(df[["sex"]]).toarray()
print(df[["sex","female","male"]])
