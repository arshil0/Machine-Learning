import numpy as np

x = np.array([34.0, 108.0, 64.0, 88.0, 99.0, 51.0, 75.0, 89.0, 112.0, 15.0, 254.0, 358.0])
y = np.array([5.0, 17.0, 11.0, 8.0, 14.0, 5.0, 5.0, 10.0, 20.0, 1.0, 20.0, 25.0])

print(sum((x - x.mean()) * (y - y.mean())) / sum())