from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


#load the data and save the corresponding features and labels in x,y variables
data = load_iris()
x, y = data.data, data.target

#split the data into training and test data (80%, 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

#let's try using logistic regression:
model = svm.SVC(C=4)

#train the model
model.fit(x_train, y_train)

#predict values for the test dataset
y_pred = model.predict(x_test)


#print the report of the model (accuracy, precision, recall)
print(classification_report(y_test, y_pred))

plt.plot(x[:2], "bo")
plt.ylabel("test")
plt.show()