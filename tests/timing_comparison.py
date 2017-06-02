import matplotlib.pyplot as plt
import sys
sys.path.append("../functions")
sys.path.append("../utils")
sys.path.append("../genetic_algorithm")
sys.path.append("../particle_swarm_optimization")

from oa_utils import optimize_settings

import genetic_algorithm
import ga_settings
import particle_swarm_optimization
import pso_settings

import ackley_function
import easom_function
import griewank_function
import rosenbrock_function

def num_parts_vs_time(o_algorithm, settings, o_function, num_particles, plot=False):
    algorithm_name = o_algorithm.get_name()
    func_name = o_function.func_globals['__name__'].replace('_', ' ').title()

    times = []
    accuracy = []

    optimize_settings(settings)
    settings['num_iterations'] = 100

    for test in num_particles:
        settings['population_size'] = test

        algorithm = o_algorithm(settings, o_function)

        algorithm.start_timer()
        algorithm.run()
        algorithm.stop_timer()

        times.append(algorithm.get_time())
        accuracy.append(algorithm.get_best_x().get_fval())

    if plot:
        plt.title("%s Number of Particles vs Time on %s" % (algorithm_name, func_name))
        plt.xlabel("Number of Particles")
        plt.ylabel("Time (Seconds)")

        plt.plot(num_particles, times, 'r-')
        plt.show()

    return (times, accuracy)

def cmp_num_parts_vs_time(o_algorithm1, o_algorithm2, \
                      settings1, settings2, o_function, num_particles):
    func_name = o_function.func_globals['__name__'].replace('_', ' ').title()
    times1 = []
    times2 = []
    accuracy1 = []
    accuracy2 = []

    optimize_settings(settings1)
    optimize_settings(settings2)

    for test in num_particles:

        t, acc = num_parts_vs_time(o_algorithm1, settings1, o_function, [test], False)
        times1.append(t)
        accuracy1.append(acc)


        t, acc = num_parts_vs_time(o_algorithm2, settings2, o_function, [test], False)
        times2.append(t)
        accuracy2.append(acc)

    # timing plot
    time_fig = plt.figure()
    time_ax = time_fig.add_subplot(111)
    time_ax.set_title("GA vs. PSO timing on %s" % func_name)
    time_ax.set_xlabel("Number of Particles")
    time_ax.set_ylabel("Time (seconds)")
    time_ax.plot(num_particles, times1, 'g-')
    time_ax.plot(num_particles, times2, 'r-')
    time_ax.legend(['GA', 'PSO'])
    # accuracy plot
    acc_fig = plt.figure()
    acc_ax = acc_fig.add_subplot(111)
    acc_ax.set_title("GA vs. PSO accuracy on %s" % func_name)
    acc_ax.set_xlabel("Number of Particles")
    acc_ax.set_ylabel("Objective Function Value")
    acc_ax.plot(num_particles, accuracy1, 'g-')
    acc_ax.plot(num_particles, accuracy2, 'r-')
    acc_ax.legend(['GA', 'PSO'])

    plt.ylim(0, 1)
    plt.show()

    return (times1, times2)

def num_dims_vs_time(o_algorithm, settings, o_function, num_dims, plot=False):
    algorithm_name = o_algorithm.get_name()
    func_name = o_function.func_globals['__name__'].replace('_', ' ').title()

    times = []
    accuracy = []

    optimize_settings(settings)

    for test in num_dims:
        # create bounds
        bounds = []
        for i in range(0, test):
            bounds.append((-10,10))
        settings['number_of_dimensions'] = test
        settings['bounds'] = bounds

        algorithm = o_algorithm(settings, o_function)

        algorithm.start_timer()
        algorithm.run()
        algorithm.stop_timer()

        times.append(algorithm.get_time())
        accuracy.append(algorithm.get_best_x().get_fval())

    if plot:
        # timing plot
        time_fig = plt.figure()
        time_ax = time_fig.add_subplot(111)
        plt.title("%s Number of Dimensions vs Time on %s" % (algorithm_name, func_name))
        time_ax.set_xlabel("Number of Dimensions")
        time_ax.set_ylabel("Time (seconds)")
        time_ax.plot(num_dims, times, 'g-')
        # accuracy plot
        acc_fig = plt.figure()
        acc_ax = acc_fig.add_subplot(111)
        plt.title("%s Number of Dimensions vs Accuracy on %s" % (algorithm_name, func_name))
        acc_ax.set_xlabel("Number of Dimensions")
        acc_ax.set_ylabel("Objective Function Value")
        acc_ax.plot(num_dims, accuracy, 'g-')

        plt.show()

    return (times, accuracy)

def cmp_num_dims_vs_time(o_algorithm1, o_algorithm2, \
                      settings1, settings2, o_function, num_dims):
    func_name = o_function.func_globals['__name__'].replace('_', ' ').title()

    times1 = []
    times2 = []
    accuracy1 = []
    accuracy2 = []

    for test in num_dims:
        # test first algorithm
        t, acc = num_dims_vs_time(o_algorithm1, settings1, o_function, [test], False)
        times1.append(t)
        accuracy1.append(acc)

        # test second algorithm
        t, acc = num_dims_vs_time(o_algorithm2, settings2, o_function, [test], False)
        times2.append(t)
        accuracy2.append(acc)

    # timing plot
    time_fig = plt.figure()
    time_ax = time_fig.add_subplot(111)
    time_ax.set_title("GA vs. PSO timing on %s" % func_name)
    time_ax.set_xlabel("Number of Dimensions")
    time_ax.set_ylabel("Time (seconds)")
    time_ax.plot(num_dims, times1, 'g-')
    time_ax.plot(num_dims, times2, 'r-')
    time_ax.legend(['GA', 'PSO'])
    # accuracy plot
    acc_fig = plt.figure()
    acc_ax = acc_fig.add_subplot(111)
    acc_ax.set_title("GA vs. PSO accuracy on %s" % func_name)
    acc_ax.set_xlabel("Number of Dimensions")
    acc_ax.set_ylabel("Objective Function Value")
    acc_ax.plot(num_dims, accuracy1, 'g-')
    acc_ax.plot(num_dims, accuracy2, 'r-')
    acc_ax.legend(['GA', 'PSO'])

    plt.show()

    return (times1, times2)

if __name__ == "__main__":
    bounds = [(-10,10), (-10,10)]

    ga_algorithm = genetic_algorithm.GA
    ga_s = ga_settings.settings
    ga_s['num_iterations'] = 100
    ga_s['num_particles'] = 50
    ga_s['number_of_dimensions'] = 2
    ga_s['bounds'] = bounds

    pso_algorithm = particle_swarm_optimization.PSO
    pso_s = pso_settings.settings
    pso_s['num_iterations'] = 100
    pso_s['num_particles'] = 50
    pso_s['number_of_dimensions'] = 2
    pso_s['bounds'] = bounds

    o_function = ackley_function.objective_function
    num_particles = [10, 20, 30, 40, 50, 100]
    #num_parts_vs_time(pso_algorithm, pso_s, o_function, num_particles, True)
    cmp_num_parts_vs_time(ga_algorithm, pso_algorithm, ga_s, pso_s, o_function, num_particles)

    sys.exit()

    num_dims = [2, 5, 10, 25, 50]
    #num_dims_vs_time(ga_algorithm, ga_s, o_function, num_dims, True)
    #num_dims_vs_time(pso_algorithm, pso_s, o_function, num_dims, True)
    cmp_num_dims_vs_time(ga_algorithm, pso_algorithm, ga_s, pso_s, o_function, num_dims)
