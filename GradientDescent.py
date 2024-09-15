import numpy as np

def gradientDescent(alpha, iterations, theta, x, y):
    m = len(x)
    for iter in range(iterations):
        predY = x.dot(theta)
        error = y - predY
        derivatives = -2/m * error.dot(x)
        theta = theta - alpha * derivatives