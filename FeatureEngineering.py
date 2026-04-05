# MEANING:
# Process of selectiong , manipulating and transforming raw data into features 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
df = fetch_openml("titanic",version= 1 , as_frame=True)["data"]
print(df.head())
df["family"] = df["sibsp"] + df["parch"]
df.loc[df["family"] > 0 ,"travelled_alone"] = 0
df.loc[df["family"] == 0 ,"travelled_alone"] = 1
df["travelled_alone"].value_counts().plot(title = "passanger travelled alone?",kind = "bar")
plt.show()
#Refer figure 4