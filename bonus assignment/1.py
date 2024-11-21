from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #Imported SVC as I need a classifier model
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

#used to draw the decision bounary
from sklearn.inspection import DecisionBoundaryDisplay

import numpy as np
import math
import random

#n is the MAX number of samples, the number of samples could be less as repeated points will be discarded
n = 100

#x will be a bunch of random samples with 2 features (so we can visualize)
x = []
y = []

#the radius of which, if the euclidean distance from the center is bigger than 4, the label is 0, otherwise 1
r = 40

#generate random samples with somewhat random binary labels (0, 1)
for i in range(n):
    #generate 2 random integers from the [0, 100]
    x1 = random.randint(0, 100)
    x2 = random.randint(0, 100)

    #if this point was already generated, discard it and pick a new one (skips a sample)
    if [x1, x2] in x:
        continue

    x.append([x1, x2])


    #85% chance to follow the rule (euclidean distance bigger than "r"), otherwise pick the opposite value (outlier / noise)
    if random.random() <= 0.85:
        if math.sqrt((50 - x1) ** 2 + (50 - x2) **2) > r:
            y.append(0)
        else:
            y.append(1)
    else:
        if x1 <= x2:
            y.append(0)
        else:
            y.append(1)


#split the data into 80% train and 20% test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#train a model that uses logistic regression
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)

#predict values for test set
y_pred_LR = logistic_regression.predict(x_test)

print("Logistic Regression performance:")
print(classification_report(y_test, y_pred_LR))

print(logistic_regression.coef_)

#NOW ONTO SVM

#Soft margin SVM
soft_margin_svm = SVC(C=1, kernel="rbf")
soft_margin_svm.fit(x_train, y_train)

y_pred_soft_SVM = soft_margin_svm.predict(x_test)


print("Soft margin SVM performance:")
print(classification_report(y_test, y_pred_soft_SVM))



#Hard margin SVM
hard_margin_svm = SVC(C=100, kernel="rbf")
hard_margin_svm.fit(x_train, y_train)

y_pred_hard_SVM = hard_margin_svm.predict(x_test)


print("Hard margin SVM performance:")
print(classification_report(y_test, y_pred_hard_SVM))



#create a dictinary to plot it later
plot_var = {}

#a nice function I found online, which breaks down an array of points into different arrays for each coordinate
plot_var["x"], plot_var["y"] = zip(*x)

#create a "color" array in the dictionary to draw each labeled point with a different color
plot_var["c"] = []

for value in y:
    if value == 0:
        plot_var["c"].append("blue")
    else:
        plot_var["c"].append("green")

#let's draw a plot for each model
figure, sub_plt = plt.subplots(1, 3, figsize=(10, 10), )

#lOGISTIC REGRESSION PLOT:
#draw a scatterplot of the points so we can see what's happening, coloring 0 labeled points blue and 1 labeled points green
sub_plt[0].scatter(plot_var["x"], plot_var["y"], c=plot_var["c"])
sub_plt[0].set_aspect('equal', 'box')
sub_plt[0].set(xlim=(-3, 103), ylim=(-3, 103))
sub_plt[0].set_title("Logistic regression")
sub_plt[0].set_xlabel(f"Test Accuracy: {accuracy_score(y_pred_LR, y_test)}")

#get the coefficients and y-intercept to plot the decision boundary
lr_coef = logistic_regression.coef_[0]
lr_intercept = logistic_regression.intercept_

# Decision boundary: θ0 + θ1*x1 + θ2*x2 = 0
# Rearrange to x2 = -(θ0 + θ1*x1) / θ2

#set up the x-axis from min to max
x_vals = np.linspace(np.array(([i[0] for i in x])).min(), np.array(([i[0] for i in x])).max(), 100)
#plot the y values for each x to get the line
y_vals = -(lr_intercept + lr_coef[0] * x_vals) / lr_coef[1]

sub_plt[0].plot(x_vals, y_vals, 'k--', c="r", label='Decision Boundary')


#SVM (Soft margin):
sub_plt[1].scatter(plot_var["x"], plot_var["y"], c=plot_var["c"])
sub_plt[1].set_aspect('equal', 'box')
sub_plt[1].set(xlim=(-3, 103), ylim=(-3, 103))
sub_plt[1].set_title("Soft margin SVM (C = 1)")
sub_plt[1].set_xlabel(f"Test Accuracy: {accuracy_score(y_pred_soft_SVM, y_test)}")

x = np.array(x)

#shamelessly copy pasting the code from the sklearn documentation
# Plot decision boundary and margins
common_params = {"estimator": soft_margin_svm, "X": x, "ax": sub_plt[1]}
DecisionBoundaryDisplay.from_estimator(
    **common_params,
    response_method="predict",
    plot_method="pcolormesh",
    alpha=0.3,
)
DecisionBoundaryDisplay.from_estimator(
    **common_params,
    response_method="decision_function",
    plot_method="contour",
    levels=[-1, 0, 1],
    colors=["k", "k", "k"],
    linestyles=["--", "-", "--"],
)

# Plot bigger circles around samples that serve as support vectors
sub_plt[1].scatter(
    soft_margin_svm.support_vectors_[:, 0],
    soft_margin_svm.support_vectors_[:, 1],
    s=150,
    facecolors="none",
    edgecolors="k",
)


#SVM (Hard margin):
sub_plt[2].scatter(plot_var["x"], plot_var["y"], c=plot_var["c"])
sub_plt[2].set_aspect('equal', 'box')
sub_plt[2].set(xlim=(-3, 103), ylim=(-3, 103))
sub_plt[2].set_title("Hard margin SVM ( C = 100)")
sub_plt[2].set_xlabel(f"Test Accuracy: {accuracy_score(y_pred_hard_SVM, y_test)}")

#shamelessly copy pasting the code from the sklearn documentation (again)
# Plot decision boundary and margins
common_params = {"estimator": soft_margin_svm, "X": x, "ax": sub_plt[2]}
DecisionBoundaryDisplay.from_estimator(
    **common_params,
    response_method="predict",
    plot_method="pcolormesh",
    alpha=0.3,
)
DecisionBoundaryDisplay.from_estimator(
    **common_params,
    response_method="decision_function",
    plot_method="contour",
    levels=[-1, 0, 1],
    colors=["k", "k", "k"],
    linestyles=["--", "-", "--"],
)

# Plot bigger circles around samples that serve as support vectors
sub_plt[2].scatter(
    soft_margin_svm.support_vectors_[:, 0],
    soft_margin_svm.support_vectors_[:, 1],
    s=150,
    facecolors="none",
    edgecolors="k",
)

plt.show()