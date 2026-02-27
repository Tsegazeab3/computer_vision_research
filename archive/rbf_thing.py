#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# Original data
data = np.array([
    [0.5, 90], [1.5, 60], [2.5, 50], [3.25, 30], [6.5, 15], [9.75, 7.5],
    [13, 4.5], [20, 3.2], [26, 2.8], [36, 2.4], [45, 2.2], [54, 1.8],
    [67, 1.5], [80, 1.3], [90, 1.1], [110, 1.05], [130, 0.9],
    [145, 0.65], [155, 0.56], [160, 0.51], [200, 0.4881704]
])

x = data[:, 0]
y = data[:, 1]

# Fit RBF
rbf = Rbf(x, y, function='linear')

# Dense x for plotting
x_dense = np.linspace(x.min(), x.max(), 1000)
y_dense = rbf(x_dense)

# Plot
plt.figure()
plt.plot(x_dense, y_dense)
plt.scatter(x, y)
plt.xlabel("avg_PPP")
plt.ylabel("Gain")
plt.title("RBF Interpolation (linear)")
plt.show()

