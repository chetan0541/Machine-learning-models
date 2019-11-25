import matplotlib.pyplot as matplotlib
import numpy as np



def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return res

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)
matplotlib.plot(z , phi_z)
matplotlib.axvline(0, color='k')
matplotlib.ylim(-0.1, 1.1)
matplotlib.xlabel('z')
matplotlib.ylabel('$\phi (y)$')
matplotlib.yticks([0, 0.5, 1])
ax = matplotlib.gca()
ax.yaxis.grid(True)
matplotlib.show()