from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../functions')

from easom_function import objective_function

fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-10, 10, .1)
Y = np.arange(-10, 10, .1)
#X = np.arange(-1.5, 1.5, .05)
#Y = np.arange(-0.5, 2, .05)
X, Y = np.meshgrid(X, Y)
params = [X, Y]
Z = objective_function(params)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                       linewidth=0, antialiased=False)
ax.contour(X, Y, Z, zdir='z', offset=-1, cmap=cm.jet)
#ax.set_zlim(0, 1000)

#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.title("Easom Function")
plt.xlabel("x1")
plt.ylabel("x2")
ax.set_zlabel("F(x1, x2)")
plt.show()
