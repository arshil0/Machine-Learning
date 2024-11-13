from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

#I need to import the data in python, to do that I will use numeric values (0, 1, 2) as classes so it's easier for python to fit the data

#age: youth = 0, middle_aged = 1, senior = 2
#income: low = 0, medium = 1, high = 2
#student: no = 0, yes = 1
#credit_rating: fair = 0, excelent = 1
#buys_computer: no = 0, yes = 1

#I will use the first 10 samples for training, last 4 for testing (hopefully they are correct)

x_train = [
    [0, 2, 0, 0],
    [0, 2, 0, 1],
    [1, 2, 0, 0],
    [2, 1, 0, 0],
    [2, 0, 1, 0],
    [2, 0, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [2, 1, 1, 0]
]

y_train = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]

x_test = [
    [0, 1, 1, 1],
    [1, 1, 0, 1],
    [1, 2, 1, 0],
    [2, 1, 0, 1]
]

y_test = [1, 1, 1, 0]

#a)

#create the model
decision_tree = DecisionTreeClassifier(criterion="gini")

#train the model on the training data
decision_tree.fit(x_train, y_train)

#predict values for test data
y_pred = decision_tree.predict(x_test)

print("DECISION TREE:")

print(f"Predicted Y values: {y_pred}")
print(f"Actual Y values: {y_test}")

print(f"Accuracy: {accuracy_score(y_pred=y_pred, y_true= y_test)}")

#for some reason when I ran this the first time the accuracy was 0.75, but after that it's 1.0 
#I changed nothing in the code


#b)
print("\n\n")
#posterior probabilities will be:
#P(buys computer = yes) = 6 / 10
#P(buys computer = no) = 4 / 10

#create the model
nayive_bayes = GaussianNB()

#fit the training data
nayive_bayes.fit(x_train, y_train)

#predict values for test data
y_pred = nayive_bayes.predict(x_test)

print("NAIVE BAYES:")

print(f"Predicted Y values: {y_pred}")
print(f"Actual Y values: {y_test}")

print(f"Accuracy: {accuracy_score(y_pred=y_pred, y_true= y_test)}")

#pretty bad accuracy


#c)

#Decision tree seems to be doing better than Naive Bayes, but again, we can't say much as our sample size is extremely small

#I should've probably wrote this on paper but here we go.

#when we use RIPPER, the most frequent label class becomes the default and we derive rules for less frequent classes.
#Buys_computer = yes is more frequent, so we pick buys_computer = no

#I will go sample by sample from top to bottom for samples with label "No":
#if(youth, high, no, fair) = no
#if(youth, high, no) = no

#here we will have 2 conditions (rules)
#if(youth, high, no OR senior, low, yes, excelent) = no

#for the 8'th sample, we can remove "high", as youth and no seem to result in "no"
#if(youth, no OR senior, low, yes, excelent) = no

#that is our final rule, let's test it for the 4 test sets.

#sample 11:
#none of the conditions are satisfied, so prediction will be "YES", which is true!

#sample 12:
#none of the conditions are satisfied, so prediction will be "YES", which is true!

#sample 13:
#none of the conditions are satisfied, so prediction will be "YES", which is true!

#sample 14:
#none of the conditions are satisfied, so prediction will be "YES", which is WRONG!

#the RIPPER model predicted yes on every sample, so the rules may be too specific

#if we adjust our rule to remove extra conditions.
#if(youth, no OR senior, excelent) = no

#we can get a 100% accuracy on test set!

