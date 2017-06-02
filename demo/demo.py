import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

import sys
sys.path.append("../utils")
sys.path.append("../functions")
sys.path.append("../particle_swarm_optimization")
sys.path.append("../genetic_algorithm")

import oa_utils
from oa_utils import read_xy_data
from regression_utils import get_regression_coef

from particle_swarm_optimization import PSO
import pso_settings

from genetic_algorithm import GA
import ackley_function
import ga_ackley_sparse_settings, ga_ackley_dense_settings
import pso_ackley_sparse_settings, pso_ackley_dense_settings
import easom_function
import ga_easom_sparse_settings, ga_easom_dense_settings, ga_easom_dense_settings1
import griewank_function
#import ga_griewank_sparse_settings, ga_griewank_dense_settings
import pso_griewank_sparse_settings, pso_griewank_dense_settings


colors = ['blue', 'green', 'red', 'magenta', 'yellow', 'black']

def plot_data(ax, mma_arr, func, zbounds=None):
    c_i = 0
    to_plot = mma_arr
    algorithm_name = "Genetic Algorithm"

    x1_name = 'Selection Cutoff'
    x2_name = 'Mutation Rate'
    x1_start = 0.1
    x1_step = 0.1
    x1_end = 1.0
    x2_start = 0.1
    x2_step = 0.1
    x2_end = 1.0

    func_file = func + '_function'
    func_name = func + ' function'
    func_name = func_name.title()
    title = 'Response Surface of %s vs %s of %s on %s' % (x1_name, x2_name, \
                                                          algorithm_name, func_name)

    if zbounds is not None:
        ax.set_zlim(zbounds[0], zbounds[1])

    for j in to_plot:
        f = '%s/cutoff_vs_rate_(mma_%.2f)_' % (func, j)
        filename = '../data/ga/' + f + func_file + '.dat'
        #filename = input_loc + gen_filename(x1_var, x2_var, func)
        (X, y) = read_xy_data(filename)

        b = get_regression_coef(X, y)

        pltx = np.arange(x1_start, x1_end+x1_step, x1_step)
        plty = np.arange(x2_start, x2_end+x2_step, x2_step)
        pltX, pltY = np.meshgrid(pltx, plty)
        F = b[0] + b[1]*pltX + b[2]*pltY + b[3]*pltX*pltX + b[4]*pltX*pltY + b[5]*pltY*pltY
        ax.plot_wireframe(pltX, pltY, F, color=colors[c_i])

        c_i += 1
        if c_i > 6:
            c_i = 0

    ax.legend(to_plot, title='Max Allowed Mutation Size', loc='lower left')
    ax.set_title(title)
    ax.set_xlabel('Selection Cutoff')
    ax.set_ylabel('Mutation Rate')
    ax.set_zlabel('Median Euclidean Distance from Global Minimum')

def run_algorithm(algorithm, settings, func, reset=False):
    c = ''
    settings['plot'] = True

    while c != 'n' and c != 'b' and c != 'q' and not (c <= '9' and c >= '0'):
        if reset:
            settings['cg'] = 0.0
        a = algorithm(settings, func)
        c = oa_utils.get_char()
        if c == 'n' or c == 'b' or c == 'q' or (c <= '9' and c >= '0'):
            return c
        a.run()
        print(str(a))
        c = oa_utils.get_char()
    return c

def plot_func(name, func, ax):
    name = name.title()

    func_ax.clear()
    X = np.arange(-10, 10, .3)
    Y = np.arange(-10, 10, .3)
    X, Y = np.meshgrid(X, Y)
    params = [X, Y]
    Z = func(params)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                                linewidth=0, antialiased=False)
    ax.contour(X, Y, Z, zdir='z', offset=-1, cmap=cm.jet)
    ax.set_title("%s Function" % name)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("F(x1, x2)")

if __name__ == "__main__":
    plt.ion()

    #response_surface_fig = plt.figure()
    #ax = response_surface_fig.add_subplot(111, projection='3d')

    func_fig = plt.figure()
    func_ax = func_fig.add_subplot(111, projection='3d')

    c = ''
    case = 0
    while case != 9:
        if case == 0:
            c = oa_utils.get_char()

        # ackley
        elif case == 1:
            print("Explore Ackley with GA")
            # response surface
            #func = 'ackley'
            #zbounds = (0,6)
            #mma_arr = [0.1, 0.5, 0.9]
            #plot_data(ax, mma_arr, func, zbounds)
            # function visual
            plot_func('ackley', ackley_function.objective_function, func_ax)
            # algorithm
            c = run_algorithm(GA, ga_ackley_sparse_settings.settings, ackley_function.objective_function)
            #ax.clear()
        elif case == 2:
            print("Investigate Ackley with GA")
            plot_func('ackley', ackley_function.objective_function, func_ax)
            c = run_algorithm(GA, ga_ackley_dense_settings.settings, ackley_function.objective_function)
            #func = 'ackley'
            #zbounds = (0,6)
            #mma_arr = [0.1, 0.5, 0.9]
            #plot_data(ax, mma_arr, func, zbounds)
            #c = run_algorithm(GA, ga_bad_ackley_settings.settings, ackley_function.objective_function)
            #ax.clear()
        elif case == 3:
            print("Explore Ackley with PSO")
            plot_func('ackley', ackley_function.objective_function, func_ax)
            c = run_algorithm(PSO, pso_ackley_sparse_settings.settings, ackley_function.objective_function)
        elif case == 4:
            print("Investigate Ackley with PSO")
            plot_func('ackley', ackley_function.objective_function, func_ax)
            c = run_algorithm(PSO, pso_ackley_dense_settings.settings, ackley_function.objective_function)

        # easom
        elif case == 5:
            print("Investigating Easom too early with GA")
            plot_func('easom', easom_function.objective_function, func_ax)
            c = run_algorithm(GA, ga_easom_dense_settings.settings, easom_function.objective_function)
        elif case == 6:
            print("Explore Easom with GA")
            plot_func('easom', easom_function.objective_function, func_ax)
            # response surface
            #func = 'easom'
            #zbounds = (0,3)
            #mma_arr = [0.1, 0.5, 0.9]
            #plot_data(ax, mma_arr, func, zbounds)
            # function visual
            # algorithm
            c = run_algorithm(GA, ga_easom_sparse_settings.settings, easom_function.objective_function)
            #ax.clear()
        elif case == 7:
            print("Investigate Easom with GA")
            plot_func('easom', easom_function.objective_function, func_ax)
            #func = 'easom'
            #zbounds = (0,3)
            #mma_arr = [0.1, 0.5, 0.9]
            #plot_data(ax, mma_arr, func, zbounds)
            c = run_algorithm(GA, ga_easom_dense_settings1.settings, easom_function.objective_function)
            #ax.clear()

        # griewank
        elif case == 8:
            print("Narrow in on Griewank minimum with PSO")
            plot_func('griewank', griewank_function.objective_function, func_ax)
            # response surface
            #mma_arr = [0.3, 0.5, 0.9]
            #func = 'griewank'
            #zbounds = (5,9)
            #plot_data(ax, mma_arr, func, zbounds)
            # function visual
            func_ax.clear()
            X = np.arange(-10, 10, .3)
            Y = np.arange(-10, 10, .3)
            X, Y = np.meshgrid(X, Y)
            params = [X, Y]
            Z = griewank_function.objective_function(params)
            surf = func_ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                                    linewidth=0, antialiased=False)
            func_ax.contour(X, Y, Z, zdir='z', offset=-1, cmap=cm.jet)
            func_ax.set_title("Griewank Function")
            func_ax.set_xlabel("x1")
            func_ax.set_ylabel("x2")
            func_ax.set_zlabel("F(x1, x2)")
            # algorithm
            c = run_algorithm(PSO, pso_griewank_sparse_settings.settings, griewank_function.objective_function, True)
            #ax.clear()


        elif case == 999:
            mma_arr = [0.3, 0.5, 0.9]
            func = 'griewank'
            zbounds = (5,9)
            plot_data(ax, mma_arr, func, zbounds)
            # function visual
            func_ax.clear()
            X = np.arange(-1.5, 1.5, .05)
            Y = np.arange(-0.5, 2, .05)
            X, Y = np.meshgrid(X, Y)
            params = [X, Y]
            Z = rosenbrock_function.objective_function(params)
            surf = func_ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
                                    linewidth=0, antialiased=False)
            func_ax.contour(X, Y, Z, zdir='z', offset=-1, cmap=cm.jet)
            func_ax.set_title("Rosenbrock Function")
            func_ax.set_xlabel("x1")
            func_ax.set_ylabel("x2")
            func_ax.set_zlabel("F(x1, x2)")
            # algorithm
            run_algorithm(GA, ga_bad_griewank_settings.settings, griewank_function.objective_function)
            ax.clear()

        # rosenbrock
        elif case == 998:
            mma_arr = [0.1, 0.5, 0.9]
            func = 'rosenbrock'
            zbounds = (0,4)
            plot_data(ax, mma_arr, func, zbounds)
            run_algorithm(GA, ga_rosenbrock_settings.settings, rosenbrock_function.objective_function)
            ax.clear()
        elif case == 997:
            mma_arr = [0.1, 0.5, 0.9]
            func = 'rosenbrock'
            zbounds = (0,4)
            plot_data(ax, mma_arr, func, zbounds)
            run_algorithm(GA, ga_bad_rosenbrock_settings.settings, rosenbrock_function.objective_function)
            ax.clear()

        if c == 'n':
            case += 1
        elif c == 'b':
            case -= 1
        elif c == 'q':
            sys.exit()
        elif c == '1' or c == '2' or c == '3' or c == '4' or c == '5' or c == '6' \
            or c == '7' or c == '8':
            case = int(c)
        if case == 0:
            case += 1
