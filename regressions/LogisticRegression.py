#Logistic regression tries to answer yes/no questions
#predY is either 0 (negative/false) or 1 (positive/true)

#our values should be between [0, 1] (inclusive)

#the formula: sigmoid(Theta * X) = 1/(1 + e**-(Theta * X))

#Cost function: -Y*ln(formula) - (1 - Y)*ln(1 - formula)

import numpy as np
import matplotlib.pyplot as plt

#defined here to be used later in gradient descent
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#almost the exact same as multiple linear regression
def gradientDescent(x, y, theta = np.array([]), alpha = 0.0001, iterations = 10000):
    #append 1 to each sample element, for theta 0
    x = np.insert(x, 0, 1, axis=1)
    independantVariables = len(x[0])
    m = len(y)
    

    #generate initial values for thetas (which is 0)
    if len(theta) == 0:
        theta = np.append(theta, [0] * independantVariables) #equivalent to the number of INDEPENDANT variables

    
    for iter in range(iterations):
        #the list of predicted values, before using the sigmoid function
        z = theta * x
        
        #create an array of predicted y values, to prepare for the next step
        predY = []
        
        #run the values of z through the sigmoid function to get the predicted values
        for value in z:
            predY.append(sigmoid(sum(value)))
        
        #prepare an array of derivatives for each theta parameter
        derivatives = []

        #iterate for each theta parameter (in my given example it's 2 theta parameters)
        for i in range(independantVariables):
            #this is the formula of the derivate of each theta
            derivatives.append(-2/m * sum(x[:,i] * (np.array(y) - np.array(predY))))
        
        #update the theta values
        theta = theta - alpha * np.array(derivatives)
    
    
    #finally let's print the cost
    cost = costFunction(y, predY)
    print('cost = ', cost)


    

    return theta



def costFunction(y, predY):
    #I don't know how to use python well enough to avoid using a for loop.
    sum = 0
    for i in range(len(y)):
        #because np.log2() doesn't like taking 0 (or close to 0) as an input, I will use an if statement here
        if y[i] == 1:
            sum -= np.log2(predY[i])
        else:
            sum -= np.log2(1 - predY[i])
    return sum


#got example from a wikipedia page about logistic regression
x = [[0.50], 
     [0.75], 
     [1.00], 
     [1.25], 
     [1.50], 
     [1.75], 
     [1.75], 
     [2.00], 
     [2.25], 
     [2.50], 
     [2.75], 
     [3.00], 
     [3.25], 
     [3.50], 
     [4.00], 
     [4.25], 
     [4.50], 
     [4.75], 
     [5.00],
     [5.50]]
y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


#run gradient descent on my given example (on wikipedia theta0 = -4.1, theta1 = 1.5, these are approximate values of course!)
print(gradientDescent(x, y, alpha=0.01, iterations=10000))

