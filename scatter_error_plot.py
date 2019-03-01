from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

matrix = np.loadtxt('alpha_batch_errors_tr.txt')

m, n = matrix.shape

fig = plt.figure()
ax = plt.axes(projection='3d')

X = matrix[:,0]
Y = matrix[:,1]
Z = matrix[:,2]

# X = np.expand_dims(matrix[:,0], axis=0)
# Y = np.expand_dims(matrix[:,1], axis=0)
# Z = np.expand_dims(matrix[:,2], axis=0)

print(np.min(Z))

ax.plot_trisurf(X,Y,Z)

plt.show()
