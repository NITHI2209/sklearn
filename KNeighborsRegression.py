import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
X,y=load_iris(return_X_y=True)
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X,y)
pred = model.predict(X)
plt.scatter(pred,y)
plt.show()
#Refer Figure 2