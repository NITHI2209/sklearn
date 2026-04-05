import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
df = fetch_openml("titanic",version= 1 , as_frame=True)["data"]
# df.info()
print(df.isnull().sum())
sns.set()
miss_val_per = pd.DataFrame((df.isnull().sum()/len(df))*100)
miss_val_per.plot(kind="bar",title="Missing values in percentage",ylabel="percentage")
plt.show()
#Refer figure 3


