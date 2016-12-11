import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from multiprocessing import Process
from scipy.optimize import minimize

import time
import sys
sys.path.append('../genetic_algorithm')
sys.path.append('../particle_swarm_optimization')
sys.path.append('../functions')
sys.path.append('../utils')

from oa_utils import optimize_settings, write_xy_data, read_xy_data
from test_helpers import gen_filename
import regression_utils

from genetic_algorithm import GA
import ga_settings
import ga_ackley_settings
from particle_swarm_optimization import PSO
import pso_settings
import ackley_function
import easom_function
import rosenbrock_function
import griewank_function

import ga_tests

NAME  = 0
START = 1
STEP  = 2
END   = 3

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
    #if response_surface:
    #    plot = True

    # turn off settings that slow us down
    o_settings = optimize_settings(o_settings)

    func_name = o_function.func_globals['__name__']
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
            test_name = 'selection_cutoff' + '(' + str(o_settings['selection_cutoff']) + ')' + ',' + \
                        'mutation_rate' + '(' + str(o_settings['mutation_rate']) + ')' + ',' + \
                        'max_mutation_amount' + '(' + str(o_settings['max_mutation_amount']) + ')'

            print("Running test %s on %s" % (test_name, func_name))

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
                for pos in algorithm.get_best_x().pos:
                    if o_function == rosenbrock_function.objective_function:
                        squares += (pos - 1)**2
                    elif o_function == easom_function.objective_function:
                        squares += (pos - np.pi)**2
                    else:
                        squares += pos**2
                euclid_distance.append(np.sqrt(squares))

            # save histogram if true
            if save_histograms:
                hist_ax.hist(euclid_distance, hist_num_bins, range=(0, 9), normed=True)
                hist_fig.savefig(test_name + '.png')
                plt.close(hist_fig)

            # find average and save data
            #avg = sum(values)/len(values)
            #avg = median(values)
            #avg = sum(euclid_distance)/len(euclid_distance)
            avg = np.median(euclid_distance)
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

#    fname = gen_filename(x1_name, x2_name, func_name)
#    write_xy_data(X, y, fname)

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
    start_time = time.time()

    func_name = o_function.func_globals['__name__']
    tests = ga_tests.tests

    x1_start = 0.1
    x1_step = 0.1
    x1_end = 1.0
    x2_start = 0.1
    x2_step = 0.1
    x2_end = 1.0


    for k in np.arange(0.1, 1.0, 0.1):
        for t in tests.items():
            settings['max_mutation_amount'] = k

            names = t[1]
            x1_name = names['x1']
            x2_name = names['x2']
            x1_name = 'selection_cutoff'
            x2_name = 'mutation_rate'

            try:
                X, y =get_two_d_accuracy(o_algorithm, settings, o_function, \
                               x1_start, x1_step, x1_end, \
                               x2_start, x2_step, x2_end, \
                               x1_name, x2_name, \
                               population_size=20, num_tests_per_point=100, plot=False, \
                               save_histograms=False, response_surface=False \
                              )
                fname = 'cutoff_vs_rate_(mma:%d)_%s.dat' % \
                    (k, func_name)
                write_xy_data(X, y, fname)

            except Exception:
                import traceback
                print("Error ??? :(")

                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                print("*** print_exception:")
                traceback.print_exception(exc_type, exc_value, exc_traceback, \
                                      limit=2, file=sys.stdout)


    print(" === %s took %d seconds === " % (o_function.func_globals['__name__'], \
                                            time.time() - start_time))

def objective_3d_function(x):
    b = [ 3.44203965, 0.67407314, -2.50195918, -2.4948652, -1.51992894, \
          0.64943886, 0.36614819, 1.05027015, 0.76315575, 1.22643218 ]
    return b[0] + \
        b[1]*x[0] + \
        b[2]*x[1] + \
        b[3]*x[2] + \
        b[4]*x[0]*x[0] + \
        b[5]*x[0]*x[1] + \
        b[6]*x[0]*x[2] + \
        b[7]*x[1]*x[1] + \
        b[8]*x[1]*x[2] + \
        b[9]*x[2]*x[2]

def optimize_3d(X, y, x1_step, x2_step, x3_step):
    x1_start = min(X[:,1])
    x1_end   = max(X[:,1])
    x2_start = min(X[:,2])
    x2_end   = max(X[:,2])
    x3_start = min(X[:,3])
    x3_end   = max(X[:,3])

    # get regression coefficients
    b = regression_utils.get_regression_coef(X, y)

    opt = {'disp': False}
    sol = minimize(objective_3d_function, [0.5, 0.5, 0.5], method='SLSQP', \
                   bounds=((0,1), (0,1), (0,1)), options = opt)

    return sol

def get_3d_accuracy(o_algorithm, o_settings, o_function, \
                         x1_info, x2_info, x3_info, \
                         population_size=50, num_tests_per_point=10, \
                         save_histograms=True, debug=False):
    # turn off settings that slow us down
    o_settings = optimize_settings(o_settings)

    func_name = o_function.func_globals['__name__']
    tests = {}
    hist_num_bins = 150

    o_settings['population_size'] = population_size
    num_tests_per = num_tests_per_point

    x1_name = x1_info[NAME]
    x2_name = x2_info[NAME]
    x3_name = x3_info[NAME]
    x1_s   = x1_info[START]
    x1_e   = x1_info[END]
    x1_stp = x1_info[STEP]
    x2_s   = x2_info[START]
    x2_e   = x2_info[END]
    x2_stp = x2_info[STEP]
    x3_s   = x3_info[START]
    x3_e   = x3_info[END]
    x3_stp = x3_info[STEP]

    num_tests = int(( (int(100*x1_e)-int(100*x1_s))/int(100*x1_stp) + 1 )* \
                    ( (int(100*x2_e)-int(100*x2_s))/int(100*x2_stp) + 1 )* \
                    ( (int(100*x3_e)-int(100*x3_s))/int(100*x3_stp) + 1 ) )
    X = np.ones(shape=(num_tests,10))
    y = np.zeros(shape=(num_tests,1))

    n = 0

    for i in np.arange(x1_s, x1_e+x1_stp, x1_stp):
        for j in np.arange(x2_s, x2_e+x2_stp, x2_stp):
            for k in np.arange(x3_s, x3_e+x3_stp, x3_stp):
                # set settings for this test
                o_settings[x1_name] = i
                o_settings[x2_name] = j
                o_settings[x3_name] = k

                # initial variables
                values = []
                euclid_distance = []
                test_name = x1_name + '(' + str(i) + ')' + ',' + \
                            x2_name + '(' + str(j) + ')' + ',' + \
                            x3_name + '(' + str(k) + ')'

                print("Running test %s" % test_name)

                # create histogram plot if true
                if save_histograms:
                    hist_fig = plt.figure()
                    hist_ax = hist_fig.add_subplot(111)

                # run optimization algorithm
                for t in range(0,num_tests_per):
                    algorithm = o_algorithm(o_settings, o_function)
                    algorithm.run()
                    # save enf values
                    values.append(algorithm.get_best_x().get_fval())
                    # euclidean distance
                    squares = 0
                    for pos in algorithm.get_best_x().pos:
                        if o_function == rosenbrock_function.objective_function:
                            squares += (pos - 1)**2
                        elif o_function == easom_function.objective_function:
                            squares += (pos - np.pi)**2
                        else:
                            squares += pos**2
                    euclid_distance.append(np.sqrt(squares))

                # save histogram if true
                if save_histograms:
                    hist_ax.hist(values, hist_num_bins, range=(0, 1.5), normed=True)
                    hist_fig.savefig(test_name + '.png')
                    plt.close(hist_fig)

                # find average and save data
                #avg = sum(values)/len(values)
                #avg = median(values)
                #avg = sum(euclid_distance)/len(euclid_distance)
                avg = np.median(euclid_distance)
                tests[test_name] = avg

                X[n][1] = i
                X[n][2] = j
                X[n][3] = k
                X[n][4] = i*i
                X[n][5] = i*j
                X[n][6] = i*k
                X[n][7] = j*j
                X[n][8] = j*k
                X[n][9] = k*k
                y[n] = avg

                # increment test number
                n += 1

    fname = gen_filename(x1_name, x2_name, func_name)
    write_xy_data(X, y, 'ga_3d_data.dat')

    if debug:
        print("\n*** DATA ***")
        print("X")
        print(X)
        print("\ny")
        print(y)
        print("\ntests")
        print(tests)

    return (X, y)


def ga_3d_data_points(o_algorithm, settings, o_function):
    start_time = time.time()

    tests = ga_tests.tests

    x1_start = 0.1
    x1_step = 0.1
    x1_end = 1.0
    x2_start = 0.1
    x2_step = 0.1
    x2_end = 1.0
    x3_start = 0.1
    x3_step = 0.1
    x3_end = 1.0

    x1_name = 'selection_cutoff'
    x2_name = 'mutation_rate'
    x3_name = 'max_mutation_amount'

    x1_info = [x1_name, x1_start, x1_step, x1_end]
    x2_info = [x2_name, x2_start, x2_step, x2_end]
    x3_info = [x3_name, x3_start, x3_step, x3_end]

    X, y = get_3d_accuracy(o_algorithm, settings, o_function, \
                           x1_info, x2_info, x3_info, \
                           population_size=20, num_tests_per_point=10, \
                           save_histograms=False, debug=False \
                           )


    X, y = read_xy_data('ga_3d_data.dat')
    sol = optimize_3d(X, y, x1_step, x2_step, x3_step)
    print(sol)

    print(" === %s took %d seconds === " % (o_function.func_globals['__name__'], \
                                            time.time() - start_time))

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
    #ga_data_points(GA, ga_settings.settings, easom_function.objective_function)
    #sys.exit()

    # ackley
    Process( target = ga_data_points, \
            args = (GA, ga_ackley_settings.settings, ackley_function.objective_function) \
          ).start()
    # easom
    Process( target = ga_data_points, \
            args = (GA, ga_settings.settings, easom_function.objective_function) \
          ).start()
    # griewank
    Process( target = ga_data_points, \
            args = (GA, ga_settings.settings, griewank_function.objective_function) \
          ).start()
    # rosenbrock
    Process( target = ga_data_points, \
            args = (GA, ga_settings.settings, rosenbrock_function.objective_function) \
          ).start()
