import argparse # parsing command line arguments
import importlib # dynamically importing modules
import random # randint
import time # delay & timing
from math import log # used in mutation

import sys # to exit and append to path
sys.path.append('../utils')
sys.path.append('../timing')
sys.path.append('../functions')

import oa_utils # optimization algorithm utils
from timer import Timer
from plot_utils import PlotUtils # plotting each iteration if plot is True

class Organism:
    """One organsim to be used with genetic algorithm. Keeps
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
        self.fitness = self.get_fitness()

    def __str__(self):
        x_str = "["
        for x in self.pos:
            x_str += "%6.3f " % x
        x_str += "]"
        return "(id: %d, fitness: %7.4f, X: %s)" % \
                (self.id, self.fitness, x_str)

    # TODO to make this a class function with a pos parameter??
    def get_fitness(self):
        return self.func(self.pos)

class GA(Timer, object):
    """A genetic algorithm class that contains methods for handling
    the population over generations/iterations

    Attributes:
        There are not attributes for this class. All settings/attributes
        are read in from ga_settings.py which should be located in the same
        directory as this file

    NOTE: The GA methods assume the population array is sorted
    """

    def __init__(self, settings, function): # TODO add settings parameter
        super(self.__class__, self).__init__()

        # read in settings
        num_dims        = settings['number_of_dimensions']
        population_size = settings['population_size']
        bounds          = settings['bounds']

        # check to make sure num_dims and number of bounds provided match
        if len(bounds) != num_dims:
            raise ValueError("Number of dimensions doesn't match number of bounds provided")

        # set instance variables
        self.settings        = settings
        self.function        = function
        # initialize population
        self.population      = GA.__gen_population(bounds, population_size, function)
        self.total_organisms = len(self.population)
        self.best_x          = self.population[0]
        self.num_generations = 1

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
    def __gen_organism(id, bounds, function):
        # use gen_random_numbers to get a list of positions within the bounds
        return Organism(id, oa_utils.gen_random_numbers(bounds), function)

    @staticmethod
    def __gen_population(bounds, size, function):
        b = bounds
        f = function
        # generate a list of organisms
        p = [GA.__gen_organism(i+1, b, f) for i in range(0, size)]
        return GA.__sort_population(p)

    @staticmethod
    def __sort_population(p):
        return sorted(p, key=lambda o: o.fitness)

    ###########################
    ###  GA steps and loop  ###
    ###########################

    '''
    Three possible ways of doing this.
    1. have a setting that says we kill of last 20% of array or population
    2. the further you are down the array the higher your probability of dieing
    3. kill off the worst based on their distance from the best
    TODO write a test for this. simple 10 population w/ .5 cutoff test will do
    '''
    @staticmethod
    def __selection(population, cutoff, print_action=False):
        size    = len(population)
        max_f = population[0].fitness
        min_f = population[size-1].fitness

        # denominator in probability of surviving
        den = (max_f - min_f)
        if den == 0:
            print("Every organism has same objective function value.")

        for (i, organism) in enumerate(population):
            f = organism.fitness

            # check for division by zero
            if den == 0:
                normalized_f = 0
            else: # get normalized value
                normalized_f = float(f - min_f) / den

            if normalized_f > cutoff:
                # delete the organism from the population
                del population[i]

                if print_action:
                    print("Selection: Deleting organism %s" % str(organism))

        return population

    @staticmethod
    def __get_parent_index(cdf_value, arr):
        norm_sum = 0
        for i, o in enumerate(arr):
            norm_sum += o['probability']
            if norm_sum >= cdf_value:
                return i
        return -1

    @staticmethod
    def __mate_parents(id, parent1, parent2, function):
        n = len(parent1.pos)
        # randomly choose split position
        split = random.randint(0, n-1)
        # split parent positions
        pos1 = parent1.pos[0:split] + parent2.pos[split:]
        pos2 = parent2.pos[0:split] + parent1.pos[split:]
        # get id numbers
        id1 = id + 1
        id2 = id + 2
        # return the two newly created organisms
        return (Organism(id1, pos1, function), Organism(id2, pos2, function))

    """
        population: population
        size: size that the population should be after crossover
        NOTE: population must be sorted. crossover will return an unsorted
              array of the new population.
    """
    @staticmethod
    def __crossover(id, population, size, function, print_action=False):
        new_population = []
        length = len(population)
        max_f = population[length-1].fitness
        min_f = population[0].fitness

        # if size is odd
        if size % 2 == 1:
            raise ValueError("Populations with an odd size hasn't been implemented. Talk to Jesse")

        # get inversed normalized values of fitness
        # normalized value of 1 is the best. 0 is the worst
        probabilities = []
        normalized_sum = 0.0
        for o in population:
            normalized_f = (max_f - o.fitness)/(max_f - min_f)
            normalized_sum += normalized_f
            probabilities.append({'normalized_f': normalized_f})

        # calculate weight of each normalized value
        for i, p in enumerate(probabilities):
            probabilities[i]['probability'] = probabilities[i]['normalized_f']/normalized_sum

        # generate new population
        while len(new_population) < size:
            # get cdf input values
            cdf1 = random.random()
            cdf2 = random.random()
            # get index of parent from output of cdf
            i = GA.__get_parent_index(cdf1, probabilities)
            j = GA.__get_parent_index(cdf2, probabilities)
            # mate parents
            child1, child2 = GA.__mate_parents(id, population[i], population[j], function)
            id += 2
            # append children to new_population
            new_population.extend((child1, child2))

        if print_action:
            for organism in new_population:
                print("Crossover: New oganism %s" % str(organism))

        return new_population

    @staticmethod
    def __mutation(population, bounds, rate, max_mutation_amount, print_action=False):
        for organism in population:
            if random.random() < rate:
                new_pos = []
                # for each dimension
                for i in range(0, len(bounds)):
                    # take some percentage of the max mutation amount
                    x = random.uniform(0.01, 1.00)
                    delta_pos = (-1.0*log(1-x))*max_mutation_amount
                    # should we go positive or negative
                    if random.randint(0,1) == 1: delta_pos = -1.0*delta_pos
                    new_dim_pos = organism.pos[i] + delta_pos
                    # cap where we can go if we are beyond the bounds of the design space
                    if new_dim_pos < bounds[i][0]:
                        new_dim_pos = bounds[i][0]
                    elif new_dim_pos > bounds[i][1]:
                        new_dim_pos = bounds[i][1]

                    new_pos.append(new_dim_pos)

                if print_action:
                    new_pos_str = "["
                    for x in new_pos:
                        new_pos_str += "%6.3f " % x
                    new_pos_str += "]"
                    print("Mutation: Moving organism %s to %s" % \
                          (str(organism), new_pos_str))

                organism.pos = new_pos

        return population

    def __display_state(self):
        print("The best organism in generation %d is %s" \
                % (self.num_generations, str(self.get_best_x())))

    def __plot_state(self):
        pts = [(organism.pos[0], organism.pos[1]) for organism in self.population]
        self.plotutils.plot(pts)

    def __str__(self):
        return "Best Fitness: %8.4f by organism %s" % \
                (self.get_best_f(), str(self.get_best_x()))

    ####################################
    # These are the only methods that  #
    # should be called outside of this #
    # class                            #
    ####################################
    def get_best_x(self):
        return self.best_x

    def get_best_f(self):
        return self.best_x.fitness

    def do_loop(self):
        population = self.population

        population = GA.__selection(population,                        \
                                    self.settings['selection_cutoff'], \
                                    self.settings['print_actions'])

        population = GA.__crossover(self.total_organisms, \
                                    population,           \
                                    self.settings['population_size'], \
                                    self.function,        \
                                    self.settings['print_actions'])
        self.total_organisms += len(population)

        population = GA.__mutation(population, \
                                   self.settings['bounds'], \
                                   self.settings['mutation_rate'],       \
                                   self.settings['max_mutation_amount'], \
                                   self.settings['print_actions'])

        self.population = GA.__sort_population(population)
        self.num_generations += 1

        if self.population[0].fitness < self.best_x.fitness:
            self.best_x = self.population[0]

        if self.settings['plot']:
            self.__plot_state()

        if self.settings['print_iterations']:
            self.__display_state()

        if self.settings['step_through']:
            oa_utils.pause()

    def run(self):
            # iterate over generations
        while self.settings['num_iterations'] > self.num_generations:
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
        settings_module = importlib.import_module('ga_settings')
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
    ga = GA(settings, function)

    if settings['time']:
        print(" --- Initialized in %s seconds --- " % (time.time() - start_time))
        if settings['time_delay'] > 0.0 or settings['plot'] \
          or settings['print_actions'] or settings['print_iterations'] or settings['step_through']:
            print("\n --- WARNING: You are timing with either time_delay, plot, print_actions,")
            print("              print_iterations, or step_through enabled. --- \n")
            oa_utils.pause()
        ga.start_timer()
        #start_time = time.time()

    ga.run()

    if settings['time']:
        ga.stop_timer()
        print(" --- Ran for %s seconds --- " % (ga.get_time()))
        #print(" --- Ran for %s seconds --- " % (time.time() - start_time))

    # print out some data
    print("")
    print(str(ga))

    sys.exit()
