from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

matrix = np.loadtxt('alpha_errors_tr10.txt')

m, n = matrix.shape

fig = plt.figure()

X = matrix[:,0]
Y = matrix[:,1]

indices = np.argsort(X)
plt.plot(X[indices],Y[indices])

plt.show()
