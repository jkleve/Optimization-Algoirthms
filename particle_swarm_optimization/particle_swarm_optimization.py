import argparse # parsing command line arguments
import importlib # dynamically importing modules
import random # randint
import time # delay & timing
from math import sqrt
from operator import add, attrgetter
import copy

import sys # to exit and append to path
sys.path.append('../utils')
sys.path.append('../timing')

import oa_utils # optimization algorithm utils
from timer import Timer
from plot_utils import PlotUtils # plotting each iteration if plot is True

"""http://www.swarmintelligence.org/tutorials.php"""

class Particle:
    """One particle to be used with particle swarm optimization. Keeps
    track of the following attributes:

    Attributes:
        id: A number that specifies an id
        pos: An array of floats defining the organisms position is space.
        func: A function to call to calculate this organisms fitness
    """

    def __init__(self, id, pos, func):
        self.id = id
        self.pos = pos
        self.func = func
        self.velocity = [0 for b in pos]
        self.fval = self.get_fval()
        self.pbest = pos

    def __str__(self):
        x_str = "["
        for x in self.pos:
            x_str += "%6.3f " % x
        x_str += "]"
        return "(id: %d, fval: %7.4f, X: %s)" % \
                (self.id, self.fval, x_str)

    def __repr__(self):
        return "<Particle(%d)>" % self.id

    def __cmp__(self, other):
        return cmp(self.fval, other.get_fval())

    # TODO to make this a class function with a pos parameter??
    def get_fval(self):
        return self.func(self.pos)

    def get_velocity(self):
        return self.velocity

class PSO:
    """A particle swarm class that contains methods for handling
    the population over iterations

    Attributes:
        There are not attributes for this class. All settings/attributes
        are read in from pso_settings.py which should be located in the same
        directory as this file
    """

    def __init__(self, settings, function): # TODO add settings parameter
        # read in settings

        num_dims        = settings['number_of_dimensions']
        population_size = settings['population_size']
        bounds          = settings['bounds']

        if settings['velocity_type'] == 'constriction':
            phi = max(settings['cp'] + settings['cg'], 4.0)
            self.k = 2.0/abs(2.0 - phi - sqrt(phi*phi - 4.0*phi))
        else:
            self.k = 1

        # check to make sure num_dims and number of bounds provided match
        if len(bounds) != num_dims:
            raise ValueError("Number of dimensions doesn't match number of bounds provided")

        # set instance variables
        self.settings        = settings
        self.function        = function
        # initialize population
        self.population      = PSO.__gen_population(bounds, population_size, function)
        self.total_population = population_size
        self.best_x          = PSO.__get_best_particle(self.population)
        self.num_iterations = 1

        if settings['plot']:
            try:
                self.plotutils = PlotUtils(num_dims, bounds, function)
                self.__plot_state()
            except ValueError:
                print("Can not plot more than 2 dimensions")
                settings['plot'] = False

        if settings['print_iterations']:
            self.__display_state()

        if settings['step_through']:
            oa_utils.pause()

    @staticmethod
    def __gen_particle(id, bounds, function):
        # use gen_random_numbers to get a list of positions within the bounds
        return Particle(id, oa_utils.gen_random_numbers(bounds), function)

    @staticmethod
    def __gen_population(bounds, size, function):
        b = bounds
        f = function
        # generate a list of organisms
        p = [PSO.__gen_particle(i+1, b, f) for i in range(0, size)]
        return p

    ###########################
    ###  PSO steps and loop  ###
    ###########################
    @staticmethod
    def __update_velocity(population, velocity_type, print_actions, gbest, cp, cg, k, w):
        for p in population:
            if (velocity_type == 'normal'):
                p.velocity = PSO.__get_velocity(1, cp, cg, gbest, p, 1)
            elif (velocity_type == 'inertia'):
                p.velocity = PSO.__get_velocity(k, cp, cg, gbest, p, w)
            elif (velocity_type == 'constriction'):
                p.velocity = PSO.__get_velocity(k, cp, cg, gbest, p, 1)
        return population

    @staticmethod
    def __get_velocity(k, c1, c2, gbest, p, w):
        velocity_array = []
        for i, v in enumerate(p.velocity):
            velocity_array.append(k*(w*v + c1*random.random()*(p.pbest[i] - p.pos[i]) + c2*random.random()*(gbest[i] - p.pos[i])))
        return velocity_array

    @staticmethod
    def __update_position(population): # TODO put bounds on what position can be updated to
        for p in population:
            p.pos = list(map(add, p.pos, p.velocity))
            p.fval = p.get_fval()
        return population

    @staticmethod
    def __get_best_particle(population):
        return copy.deepcopy( min(population, key=attrgetter('fval')) )

    def __display_state(self):
        print("The best organism in generation %d is %s" \
                % (self.num_generations, str(self.get_best_x())))

    def __plot_state(self):
        pts = [(organism.pos[0], organism.pos[1]) for organism in self.population]
        self.plotutils.plot(pts)

    def __str__(self):
        return "Best Fitness: %8.4f by particle %s" % \
                (self.get_best_f(), str(self.get_best_x()))

    ####################################
    # These are the only methods that  #
    # should be called outside of this #
    # class                            #
    ####################################
    def get_best_x(self):
        return self.best_x

    def get_best_f(self):
        return self.best_x.fval

    def do_loop(self):
        population = self.population

        population = PSO.__update_velocity(population,              \
                                    self.settings['velocity_type'], \
                                    self.settings['print_actions'], \
                                    self.get_best_x().pos,          \
                                    self.settings['cp'],            \
                                    self.settings['cg'],            \
                                    self.k,                         \
                                    self.settings['weight'])

        population = PSO.__update_position(population)

        self.num_iterations += 1

        self.population = population

        current_best = PSO.__get_best_particle(self.population)

        if current_best.get_fval() < self.best_x.get_fval():
            self.best_x = current_best

        if self.settings['plot']:
            self.__plot_state()

        if self.settings['print_iterations']:
            self.__display_state()

        if self.settings['step_through']:
            oa_utils.pause()

    def run(self):
            # iterate over generations
        while self.settings['num_iterations'] > self.num_iterations:
            self.do_loop()
            time.sleep(self.settings['time_delay'])

########################################################################################
#                                     MAIN                                             #
########################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Accept an optional settings file')
    parser.add_argument('--settings', '-s', nargs=1, type=str, \
                        metavar='<file>', help='specify settings file to use')
    parser.add_argument('--function', '-f', nargs=1, type=str, \
                        metavar='<file>', help='specify objective function file to use')
    parser.add_argument('-v', action='store_true', help='print info when method is doing an action')
    parser.add_argument('--time', '-t', action='store_true', help='turn timing on for the algorithm')
    parser.add_argument('--plot', '-p', action='store_true', help='plot each iteration')
    args = parser.parse_args()

    function_module = None
    settings_module = None

    # get objective function
    if args.function:
        function_module = importlib.import_module(args.function[0])
    else:
        function_module = importlib.import_module('ackley_function')
    function = function_module.objective_function

    # get settings
    if args.settings:
        settings_module = importlib.import_module(args.settings[0])
    else:
        settings_module = importlib.import_module('pso_settings')
    settings = settings_module.settings

    # if -v is set change the setting
    if args.v:
        settings['print_actions'] = True
        settings['print_iterations'] = True

    # check for a couple more command line arguments
    if args.time: settings['time'] = True
    if args.plot: settings['plot'] = True

    # --- END OF ARG PARSING --- #

    # print a empty line
    print("")

    # time initialization
    if settings['time']:
        start_time = time.time()

    # create algorithm instance
    pso = PSO(settings, function)

    if settings['time']:
        print(" --- Initialized in %s seconds --- " % (time.time() - start_time))
        if settings['time_delay'] > 0.0 or settings['plot'] \
          or settings['print_actions'] or settings['print_iterations'] or settings['step_through']:
            print("\n --- WARNING: You are timing with either time_delay, plot, print_actions,")
            print("              print_iterations, or step_through enabled. --- \n")
            oa_utils.pause()
        pso.start_timer()

    # iterate over generations
    pso.run()

    if settings['time']:
        pso.stop_timer()
        print(" --- Ran for %s seconds --- " % (pso.get_time()))

    # print out some data
    print("")
    print(str(pso))

    sys.exit()
