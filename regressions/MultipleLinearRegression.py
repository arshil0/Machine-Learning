import numpy as np
import matplotlib.pyplot as plt

#travel time prediction values:
#x variables: miles traveled, number of deliveries, gas price
x = np.array([
    [89, 4, 3.84],
    [66, 1, 3.19],
    [78, 3, 3.78],
    [111, 6, 3.89],
    [44, 1, 3.57],
    [77, 3, 3.57],
    [80, 3, 3.03],
    [66, 2, 3.51],
    [109, 5, 3.54],
    [76, 3, 3.25]
])

y = np.array([7, 5.4, 6.6, 7.4, 4.8, 6.4, 7, 5.6, 7.3, 6.4])

#used for multiple linear regression
def multipleLinearRegression(x, y, theta = np.array([]), alpha = 0.0001, iterations = 10000):
    #append 1 to each sample element, for theta 0
    x = np.insert(x, 0, 1, axis=1)
    independantVariables = len(x[0])
    m = len(y)
    

    
    if len(theta) == 0:
        theta = np.append(theta, [0] * independantVariables) #equivalent to the number of INDEPENDANT variables

    print(x)
    print(theta)
    for iter in range(iterations):
        values = theta * x
        predY = []
        for value in values:
            predY.append(sum(value)) 
        
        derivatives = []
        for i in range(independantVariables):
            derivatives.append(-2/m * sum(x[:,i] * (y - predY)))
        theta = theta - alpha * np.array(derivatives)


    #copy-pasted the code below from our slides :)

    #we can't really draw an accurate graph, since we have multiple variables
    #so instead, we just draw the y values with the first independant variable, to atleast have something visual.
    #there is a high chance that the graph doesn't give much value, as some variables are not highly correlated with y as others.
    #so, we may get a bad fitting line for the specific variables
    plt.scatter(x[:, 1],y)
    plt.plot([np.min(x),np.max(x)],[min(predY), max(predY)], color = 'red')
    plt.show()

    cost = np.sum((predY - y)**2)
    meanY = np.mean(y)
    SST = np.sum((meanY - y)**2)
    SSR = SST - cost
    R_Square = (SSR/SST)*100
    print('\nSST= ', SST,'\nSSR= ', SSR, '\nSSE= ', cost, '\nR_Square= ', R_Square)

    return theta

print(multipleLinearRegression(x, y))