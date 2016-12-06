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
import rosenbrock_function


def num_parts_vs_time(o_algorithm1, o_algorithm2, \
                      settings1, settings2, o_function, num_particles):
    times1 = []
    times2 = []
    accuracy1 = []
    accuracy2 = []

    optimize_settings(settings1)
    optimize_settings(settings2)

    for test in num_particles:
        settings1['population_size'] = test

        algorithm1 = o_algorithm1(settings1, o_function)

        algorithm1.start_timer()
        algorithm1.run()
        algorithm1.stop_timer()

        times1.append(algorithm1.get_time())
        accuracy1.append(algorithm1.get_best_x().get_fval())


        settings2['population_size'] = test

        algorithm2 = o_algorithm2(settings2, o_function)

        algorithm2.start_timer()
        algorithm2.run()
        algorithm2.stop_timer()

        times2.append(algorithm2.get_time())
        accuracy2.append(algorithm2.get_best_x().get_fval())

    #plt.plot(num_particles, times1, 'g-')
    #plt.plot(num_particles, times2, 'r-')
    #plt.title("GA vs. PSO timing")
    plt.xlabel("Number of Particles")
    #plt.ylabel("Time (Seconds)")

    plt.plot(num_particles, accuracy1, 'g-')
    plt.plot(num_particles, accuracy2, 'r-')
    plt.title("GA vs. PSO accuracy")
    plt.ylabel("Objective Function Value")

    plt.legend(['GA', 'PSO'])
    plt.show()

    return (times1, times2)

if __name__ == "__main__":
    ga_algorithm = genetic_algorithm.GA
    ga_s = ga_settings.settings
    pso_algorithm = particle_swarm_optimization.PSO
    pso_s = pso_settings.settings
    o_function = easom_function.objective_function
    num_particles = [10, 50, 100, 250, 500]
    num_parts_vs_time(ga_algorithm, pso_algorithm, ga_s, pso_s, o_function, num_particles)
