import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
# print(data)
X,y = load_breast_cancer(return_X_y=True)
from sklearn.neighbors import KNeighborsClassifier
mod = KNeighborsClassifier().fit(X,y)
# mod.fit(X,y)
mod.predict(X)
prediction = mod.predict(X)
plt.scatter(prediction,y)
plt.show()
