import numpy as np
import matplotlib.pyplot as plt

#tip example
x = np.array([34.0, 108.0, 64.0, 88.0, 99.0, 51.0, 75.0, 89.0, 112.0, 15.0, 254.0, 358.0])
y = np.array([5.0, 17.0, 11.0, 8.0, 14.0, 5.0, 5.0, 10.0, 20.0, 1.0, 20.0, 25.0])


#used for simple linear regression
def simpleGradientDescent(x, y, theta = np.array([]), alpha = 0.00001, iterations = 80000):
    m = len(x)
    if len(theta) == 0:
        theta = np.append(theta, [0, 0])
    for iter in range(iterations):
        predY = theta[0] + theta[1] * x
        
        theta0Derivative = -2/m * sum(y - predY)
        theta1Derivative = -2/m * sum((y - predY) * x)
        theta[0] = theta[0] - alpha * theta0Derivative
        theta[1] = theta[1] - alpha * theta1Derivative


    #copy-pasted the code below from our slides :)
    plt.scatter(x,y)
    plt.plot([min(x),max(x)],[min(predY), max(predY)], color = 'red')
    plt.show()

    cost = np.sum((predY - y)**2)
    meanY = np.mean(y)
    SST = np.sum((meanY - y)**2)
    SSR = SST - cost
    R_Square = (SSR/SST)*100
    print("Theta 0 = ", theta[0], ": Theta 1 = ", theta[1])
    print('\nSST= ', SST,'\nSSR= ', SSR, '\nSSE= ', cost, '\nR_Square= ', R_Square)

    return theta

#print(simpleGradientDescent(x, y))



#travel time prediction values:
#x variables: miles traveled, number of deliveries, gas price
x = np.array([
    [89, 66,78, 111, 44, 77, 80, 66,109, 76],
    [4, 1, 3, 6, 1, 3, 3, 2, 5, 3],
    [3.84, 3.19, 3.78, 3.89, 3.57, 3.57, 3.03, 3.51, 3.54, 3.25]
    ])

y = np.array([7, 5.4, 6.6, 7.4, 4.8, 6.4, 7, 5.6, 7.3, 6.4])

#used for multiple linear regression
def multipleLinearRegression(x, y, theta = np.array([]), alpha = 0.00001, iterations = 10):
    independantVariables = len(x)
    m = len(y)
    print(x)
    #x = np.insert(x, 0, [1] * m) #insert a row of 1's as the first set of values, for tetha[0]
    
    
    if len(theta) == 0:
        theta = np.append(theta, [0] * independantVariables) #equivalent to the number of INDEPENDANT variables

    print(x)
    print(theta)
    for iter in range(iterations):
        predY = np.dot(theta, x)
        print(predY)
        
        derivatives = -2/m * sum(np.dot(x, (y - predY)))
        print(derivatives)
        theta = theta - alpha * derivatives


    #copy-pasted the code below from our slides :)
    plt.scatter(x,y)
    plt.plot([min(x),max(x)],[min(predY), max(predY)], color = 'red')
    plt.show()

    cost = np.sum((predY - y)**2)
    meanY = np.mean(y)
    SST = np.sum((meanY - y)**2)
    SSR = SST - cost
    R_Square = (SSR/SST)*100
    print("Theta 0 = ", theta[0], ": Theta 1 = ", theta[1])
    print('\nSST= ', SST,'\nSSR= ', SSR, '\nSSE= ', cost, '\nR_Square= ', R_Square)

    return theta

print(multipleLinearRegression(x, y))