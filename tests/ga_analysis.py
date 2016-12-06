import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sys
sys.path.append("../utils")
sys.path.append("../functions")
sys.path.append("../genetic_algorithm")

import genetic_algorithm
import ackley_function
import easom_function
import rosenbrock_function
import griewank_function
import ga_settings


from oa_utils import read_xy_data, optimize_settings
from regression_utils import get_regression_coef

def cmp_selection_cutoff_vs_mutation_rate():
    X, y = read_xy_data('selection_cutoff,mutation_rate.dat')

    b = get_regression_coef(X, y)

    x1_start = 0.3
    x1_step = 0.1
    x1_end = 0.6
    x2_start = 0.2
    x2_step = 0.1
    x2_end = 0.4

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim(0, 1)

    for i, row in enumerate(X):
        ax1.scatter(row[1],row[2],y[i])

    pltx = np.arange(x1_start, x1_end+x1_step, x1_step)
    plty = np.arange(x2_start, x2_end+x2_step, x2_step)
    pltX, pltY = np.meshgrid(pltx, plty)
    F = b[0] + b[1]*pltX + b[2]*pltY + b[3]*pltX*pltX + b[4]*pltX*pltY + b[5]*pltY*pltY
    #ax1.plot_wireframe(pltX, pltY, F)

    ax1.set_xlabel('Selection Cutoff')
    ax1.set_ylabel('Mutation Rate')
    ax1.set_zlabel('Median Objective Function Value')
    plt.show()

def cmp_func_val_over_iterations(o_algorithm, settings, o_function):
    x1_start = 0.3
    x1_step = 0.1
    x1_end = 0.6
    x2_start = 0.25
    x2_step = 0.25
    x2_end = 0.75

    x1_name = "selection_cutoff"
    x2_name = "mutation_rate"
    population_size = [50]

    optimize_settings(settings)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)

    tests1 = []
    tests2 = []
    tests3 = []

    for test in population_size:
        for i, x1 in enumerate(np.arange(x1_start, x1_end+x1_step, x1_step)):
            for j, x2 in enumerate(np.arange(x2_start, x2_end+x2_step, x2_step)):
                settings[x1_name] = x1
                settings[x2_name] = x2
                f = []

                settings['population_size'] = test

                algorithm = o_algorithm(settings, o_function)

                while settings['num_iterations'] > algorithm.num_generations:
                    f.append(algorithm.get_best_f())
                    algorithm.do_loop()

                if j == 0:
                    tests1.append("Selection Cutoff %4.2f Mutation Rate %4.2f" % (x1, x2))
                    ax1.plot(range(1,len(f)+1), f)
                elif j == 1:
                    tests2.append("Selection Cutoff %4.2f Mutation Rate %4.2f" % (x1, x2))
                    ax2.plot(range(1,len(f)+1), f)
                elif j == 2:
                    tests3.append("Selection Cutoff %4.2f Mutation Rate %4.2f" % (x1, x2))
                    ax3.plot(range(1,len(f)+1), f)

    ax1.legend(tests1)
    ax2.legend(tests2)
    ax3.legend(tests3)
    ax1.set_title('GA Comparison of Selection Cutoff & Mutation Rate on Ackley Function (50 particles)')
    ax1.set_xlabel('Number of Iterations')
    ax2.set_xlabel('Number of Iterations')
    ax3.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Objective Function Value')
    ax2.set_ylabel('Objective Function Value')
    ax3.set_ylabel('Objective Function Value')
    #ax2.ylabel('Objective Function Value')
    #ax3.ylabel('Objective Function Value')
    #plt.legend(tests)
    plt.show()

if __name__ == "__main__":
    #cmp_selection_cutoff_vs_mutation_rate()

    cmp_func_val_over_iterations(genetic_algorithm.GA, \
                                 ga_settings.settings, \
                                 ackley_function.objective_function)
