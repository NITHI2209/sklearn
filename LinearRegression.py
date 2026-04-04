import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_iris
X,y=load_iris(return_X_y=True)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)
pred = model.predict(X)
plt.scatter(pred,y)
plt.show()
# #Refer Figure 1

