import matplotlib.pyplot as plt

from particle_swarm_optimization import PSO as OptimizationAlgorithm
import pso_settings
import pso_objective_function

def num_parts_vs_time(o_algorithm, num_parts):
    tests = {}

    for test in num_parts:
        pso_settings.settings['population_size'] = test

        algorithm = o_algorithm(ga_settings.settings, ga_objective_function.objective_function)

        algorithm.start_timer()
        algorithm.run()
        algorithm.stop_timer()

        x = algorithm.get_best_x()

        print("%5d: %7.3f" % (test, algorithm.get_time()))

        tests[test] = {'time': algorithm.get_time(), 'accuracy': abs(x.get_fval())}

    return tests

def func_val_vs_iterations(o_algorithm, num_parts):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    for test in num_parts:
        f = []

        pso_settings.settings['population_size'] = test

        algorithm = o_algorithm(pso_settings.settings, pso_objective_function.objective_function)

        while pso_settings.settings['num_iterations'] > algorithm.num_iterations:
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

    num_particles = [50, 100, 500]# + range(1000, 10000, 1000)
    #tests = num_parts_vs_time(OptimizationAlgorithm, num_particles)

    func_val_vs_iterations(OptimizationAlgorithm, num_particles)
