from matplotlib import pyplot as plt
import numpy as np

np.random.seed(99)
X = np.random.rand(100)
Y = 2*X+0.5*np.random.rand(100)
plt.scatter(X, Y)
plt.show()