from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
x, y = data.data, data.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)


model = GradientBoostingRegressor()
model.fit(x_train, y_train)

predY = model.predict(x_test)

print(y_test)
print(predY)