from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets #sklearn seems to have a smaller version of MNIST

import matplotlib.pyplot as plt

#I am importing SVC because I need to use a classifier, SVR will be a regressor model.

digits = datasets.load_digits()


#this code (as much as I understood) writes the images in matrix format
#where each number is the strength of ink on that specific pixel
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#create the classifier model
model = SVC(gamma=0.0009)

# Split data into 70% train and 30% test subsets
x_train, x_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False
)

#fit the data into the model
model.fit(x_train, y_train)

#predict some values
y_pred = model.predict(x_test)


print("We used SVM to learn numbers from pixelated images, let's see its score")

print(classification_report(y_test, y_pred))

#it did pretty good!
#overall 95% accuracy (with default settings)!

#I managed to reach 97% when gamma is around 0.001, the C parameter didn't change much