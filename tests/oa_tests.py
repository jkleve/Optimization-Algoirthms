import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
#import threading
#from threading import Thread
from multiprocessing import Process
from statistics import median
import sys
sys.path.append('../genetic_algorithm')
sys.path.append('../particle_swarm_optimization')
sys.path.append('../functions')
sys.path.append('../utils')

from oa_utils import optimize_settings, write_xy_data
import regression_utils

from genetic_algorithm import GA
import ga_settings
from particle_swarm_optimization import PSO
import pso_settings
import ackley_function
import easom_function
import rosenbrock_function
import griewank_function

def gen_filename(x1_name, x2_name):
    return x1_name + ',' + x2_name + '.dat'

def num_parts_vs_time(o_algorithm, num_parts):
    tests = {}

    for test in num_parts:
        settings['population_size'] = test

        algorithm = o_algorithm(settings, objective_function)

        algorithm.start_timer()
        algorithm.run()
        algorithm.stop_timer()

        x = algorithm.get_best_x()

        print("%5d: %7.3f" % (test, algorithm.get_time()))

        tests[test] = {'time': algorithm.get_time(), 'accuracy': abs(x.get_fitness())}

    return tests

def func_val_vs_iterations(o_algorithm, num_parts):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    for test in num_parts:
        f = []

        settings.settings['population_size'] = test

        algorithm = o_algorithm(settings, objective_function)

        while settings['num_iterations'] > algorithm.num_generations:
            f.append(algorithm.get_best_f())
            algorithm.do_loop()

        ax.plot(range(1,len(f)+1), f)

    plt.title('Number of particles vs iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Objective Function Value')
    plt.legend(num_parts)
    fig_name = 'func_value_vs_iterations'
    fig.savefig(fig_name + '.png')
    plt.close(fig)

# saves all test data to file
# x1_name: name of setting in settings file
# x2_name: name of setting in settings file
def get_two_d_accuracy(o_algorithm, o_settings, o_function, \
                           x1_start, x1_step, x1_end, \
                           x2_start, x2_step, x2_end, \
                           x1_name, x2_name, \
                           population_size=50, num_tests_per_point=10, plot=True, \
                           save_histograms=True, response_surface=True, debug=False):
    if response_surface:
        plot = True

    # turn off settings that slow us down
    o_settings = optimize_settings(o_settings)

    tests = {}
    hist_num_bins = 150

    o_settings['population_size'] = population_size
    num_tests_per = num_tests_per_point
    x1_srt = x1_start
    x1_e   = x1_end
    x1_stp = x1_step
    x2_srt = x2_start
    x2_e   = x2_end
    x2_stp = x2_step
    num_tests = int(( (int(100*x1_e)-int(100*x1_srt))/int(100*x1_stp) + 1 )* \
                    ( (int(100*x2_e)-int(100*x2_srt))/int(100*x2_stp) + 1 ))
    X = np.ones(shape=(num_tests,6))
    y = np.zeros(shape=(num_tests,1))

    n = 0

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        if o_function != griewank_function.objective_function:
            ax1.set_zlim(0, 1)

    for i in np.arange(x1_srt, x1_e+x1_stp, x1_stp):
        for j in np.arange(x2_srt, x2_e+x2_stp, x2_stp):
            # set settings for this test
            o_settings[x1_name] = i
            o_settings[x2_name] = j

            # initial variables
            values = []
            euclid_distance = []
            test_name = x1_name + '(' + str(i) + ')'+ ',' + x2_name + '(' + str(j) + ')'

            print("Running test %s" % test_name)

            # create histogram plot if true
            if save_histograms:
                hist_fig = plt.figure()
                hist_ax = hist_fig.add_subplot(111)

            # run optimization algorithm
            for k in range(0,num_tests_per):
                algorithm = o_algorithm(o_settings, o_function)
                algorithm.run()
                # save enf values
                values.append(algorithm.get_best_x().get_fval())
                # euclidean distance
                squares = 0
                for square in algorithm.get_best_x().pos:
                    if o_function == rosenbrock_function.objective_function:
                        squares += (square-1)**2
                    elif o_function == easom_function.objective_function:
                        squares += (square-np.pi)**2
                    else:
                        squares += square**2
                euclid_distance.append(np.sqrt(squares))

            # save histogram if true
            if save_histograms:
                hist_ax.hist(values, hist_num_bins, range=(0, 1.5), normed=True)
                hist_fig.savefig(test_name + '.png')
                plt.close(hist_fig)

            # find average and save data
            #avg = sum(values)/len(values)
            #avg = median(values)
            avg = median(euclid_distance)
            tests[test_name] = avg
            if plot:
                ax1.scatter(i,j,avg)

            X[n][1] = i
            X[n][2] = j
            X[n][3] = i*i
            X[n][4] = i*j
            X[n][5] = j*j
            y[n] = avg

            # increment test number
            n += 1

    fname = gen_filename(x1_name, x2_name)
    write_xy_data(X, y, fname)

    if debug:
        print("\n*** DATA ***")
        print("X")
        print(X)
        print("\ny")
        print(y)
        print("\ntests")
        print(tests)

    if response_surface:
        # get regression coefficients
        b = regression_utils.get_regression_coef(X, y)

        pltx = np.arange(x1_start, x1_end+x1_step, x1_step)
        plty = np.arange(x2_start, x2_end+x2_step, x2_step)
        pltX, pltY = np.meshgrid(pltx, plty)
        F = b[0] + b[1]*pltX + b[2]*pltY + b[3]*pltX*pltX + b[4]*pltX*pltY + b[5]*pltY*pltY
        ax1.plot_wireframe(pltX, pltY, F)

    if plot:
        print("\nPlotting ...")
        x1_name = x1_name[0].upper() + x1_name[1:]
        x2_name = x2_name[0].upper() + x2_name[1:]
        ax1.set_xlabel(x1_name)
        ax1.set_ylabel(x2_name)
        ax1.set_zlabel('Average Euclidean Distance from Global Minimum')
        plt.show()

    return (X, y)

def ga_data_points(o_algorithm, settings, o_function):
    x1_start = 0.1
    x1_step = 0.1
    x1_end = 1.0
    x2_start = 0.1
    x2_step = 0.1
    x2_end = 1.0
    # x1_start = 0.5
    # x1_step = 0.1
    # x1_end = 0.9
    # x2_start = 0.6
    # x2_step = 0.1
    # x2_end = 0.9
    x1_name = "selection_cutoff"
    x2_name = "mutation_rate"
    # x1_start = 0.5
    # x1_step = 0.5
    # x1_end = 1.0
    # x2_start = 0.1
    # x2_step = 0.1
    # x2_end = 0.4
    # x1_name = "max_mutation_amount"
    # x2_name = "mutation_rate"

    return get_two_d_accuracy(o_algorithm, settings, o_function, \
                              x1_start, x1_step, x1_end, \
                              x2_start, x2_step, x2_end, \
                              x1_name, x2_name, \
                              population_size=50, num_tests_per_point=10, plot=True, \
                              save_histograms=False, response_surface=False \
                             )

def pso_data_points(o_algorithm, settings, o_function):
    x1_start = 0.0
    x1_step = 0.1
    x1_end = 0.4
    x2_start = 0.6
    x2_step = 0.1
    x2_end = 0.9
    x1_name = "cp"
    x2_name = "cg"

    return get_two_d_accuracy(o_algorithm, settings, o_function, \
                              x1_start, x1_step, x1_end, \
                              x2_start, x2_step, x2_end, \
                              x1_name, x2_name, \
                              population_size=50, num_tests_per_point=50, plot=True, \
                              save_histograms=False, response_surface=True \
                             )

if __name__ == "__main__":

    #num_particles = [50, 100, 500]# + range(1000, 10000, 1000)
    #tests = num_parts_vs_time(OptimizationAlgorithm, num_particles)

    #func_val_vs_iterations(OptimizationAlgorithm, num_particles)

    #pso_data_points(PSO, pso_settings.settings, rosenbrock_function.objective_function)
    ga_data_points(GA, ga_settings.settings, griewank_function.objective_function)
    sys.exit()

    Process( target = ga_data_points, \
            args = (GA, ga_settings.settings, rosenbrock_function.objective_function) \
          ).start()

    Process( target = pso_data_points, \
            args = (PSO, pso_settings.settings, rosenbrock_function.objective_function) \
          ).start()
