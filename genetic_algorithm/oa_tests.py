import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from statistics import median

from genetic_algorithm import GA as OptimizationAlgorithm
from ackley_settings import settings
from ackley_function import objective_function

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
    plt.ylabel('Number of Particles')
    plt.legend(num_parts)
    fig_name = 'particles_vs_iterations'
    fig.savefig(fig_name + '.png')
    plt.close(fig)

if __name__ == "__main__":

    #num_particles = [50, 100, 500]# + range(1000, 10000, 1000)
    #tests = num_parts_vs_time(OptimizationAlgorithm, num_particles)

    #func_val_vs_iterations(OptimizationAlgorithm, num_particles)

    tests = {}


    # selection cutoff [0.1, 0.9]
    # mutation rate    [0.05, 0.8]
    # number of runs 50 w/ 50 particles
    settings['population_size'] = 50
    num_tests_per = 50
    sc_start = 0.1
    sc_end   = 0.9
    sc_step  = 0.1
    mt_start = 0.1
    mt_end   = 0.8
    mt_step  = 0.1
    num_tests = ( (sc_end-sc_start)/sc_step + 1 )*( (mt_end-mt_start)/mt_step + 1 )
    X = np.ones(shape=(num_tests,6))
    y = np.zeros(shape=(num_tests,1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in np.arange(sc_start, sc_end+sc_step, sc_step):
        for j in np.arange(mt_start, mt_end+mt_step, mt_step):
            settings['selection_cutoff'] = i
            settings['mutation_rate'] = j

            values = []
            test_name = str(i) + ',' + str(j)
            print("Running test %s" % test_name)

            for k in range(0,num_tests_per):
                algorithm = OptimizationAlgorithm(settings, objective_function)
                algorithm.run()

                values.append(algorithm.get_best_x().get_fitness())
                X[k][1] = i
                X[k][2] = j
                X[k][3] = i**i
                X[k][4] = i**j
                X[k][5] = j**j
                y[k] = algorithm.get_best_x().get_fitness()

            #avg = sum(values)/len(values)
            avg = median(values)
            tests[test_name] = avg
            ax.scatter(i,j,avg)

    XtX = np.matmul(X.transpose(), X)
    XtXinv = np.linalg.inv(XtX)
    XtXinvXt = np.matmul(XtXinv, X.transpose())

    b = np.matmul(XtXinvXt, y)

    print(b)

    print(tests)
    ax.set_xlabel('Selection Cutoff')
    ax.set_ylabel('Mutation Rate')
    ax.set_zlabel('Averaged Objective Function Value')
    #plt.show()
