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

print(simpleGradientDescent(x, y))
