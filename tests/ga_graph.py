import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sys
sys.path.append("../utils")

from oa_utils import read_xy_data
from test_helpers import gen_filename
from regression_utils import get_regression_coef

input_loc = '../data/ga/'
output_loc = '../tmp/'
algorithm_name = 'GA'

NAME  = 0
START = 1
STEP  = 2
END   = 3

colors = ['blue', 'green', 'red', 'magenta', 'yellow', 'black']

def plot_data(x1_info, x2_info, func, zbounds=None):
    c_i = 0
    to_plot = [0.1, 0.5, 0.9]

    x1_var = x1_info[NAME]
    x2_var = x2_info[NAME]
    x1_name = x1_var.replace('_', ' ').title()
    x2_name = x2_var.replace('_', ' ').title()

    func = func + '_function'
    func_name = func.replace('_', ' ').title()
    title = 'Response Surface of %s vs %s of %s on %s' % (x1_name, x2_name, \
                                                          algorithm_name, func_name)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    if zbounds is not None:
        ax1.set_zlim(zbounds[0], zbounds[1])

    for j in to_plot:
        f = 'rosenbrock/cutoff_vs_rate_(mma_%.2f)_' % j
        filename = input_loc + f + func + '.dat'
        #filename = input_loc + gen_filename(x1_var, x2_var, func)
        (X, y) = read_xy_data(filename)

        b = get_regression_coef(X, y)

        #for i, row in enumerate(X):
        #    ax1.scatter(row[1],row[2],y[i], color=colors[c_i])

        pltx = np.arange(x1_start, x1_end+x1_step, x1_step)
        plty = np.arange(x2_start, x2_end+x2_step, x2_step)
        pltX, pltY = np.meshgrid(pltx, plty)
        F = b[0] + b[1]*pltX + b[2]*pltY + b[3]*pltX*pltX + b[4]*pltX*pltY + b[5]*pltY*pltY
        ax1.plot_wireframe(pltX, pltY, F, color=colors[c_i])
        #ax1.contour(pltX, pltY, F, zdir='z', offset=0, cmap=cm.jet)

        c_i += 1
        if c_i > 6:
            c_i = 0

    ax1.set_title(title)
    ax1.set_xlabel(x1_name)
    ax1.set_ylabel(x2_name)
    ax1.set_zlabel('Median Euclidean Distance from Global Minimum')
    ax1.legend(to_plot, title='Max Allowed Mutation Size', loc='lower left')
    plt.show()

if __name__ == "__main__":

    # user inputs
    func = 'rosenbrock'
    x1_name = 'selection_cutoff'
    x2_name = 'mutation_rate'
    x1_start = 0.1
    x1_step = 0.1
    x1_end = 1.0
    x2_start = 0.1
    x2_step = 0.1
    x2_end = 1.0
    zbounds = (0,7)

    # don't touch
    x1_info = [x1_name, x1_start, x1_step, x1_end]
    x2_info = [x2_name, x2_start, x2_step, x2_end]

    plot_data(x1_info, x2_info, func, zbounds)
