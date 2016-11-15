import random # randint
import time # delay
from math import log

import sys # to exit and append to path
sys.path.append('../utils')

import oa_utils # optimization algorithm utils
from plot_utils import PlotUtils
from ga_settings import settings
from ga_objective_function import objective_function

class Organism:
    """One organsim to be used with genetic algorithm. Keeps
    track of the following attributes:

    Attributes:
        id: A number that specifies an id
        pos: An array of floats defining the organisms position is space.
        func: A function to call to calculate this organisms fitness
    """

    def __init__(self, id, pos, func=objective_function):
        self.id = id
        self.pos = pos
        self.func = func
        self.fitness = self.get_fitness()

    # TODO to make this a class function with a pos parameter??
    def get_fitness(self):
        return self.func(self.pos)

class GA:
    """A genetic algorithm class that contains methods for handling
    the population over generations/iterations

    Attributes:
        There are not attributes for this class. All settings/attributes
        are read in from ga_settings.py which should be located in the same
        directory as this file

    NOTE: The GA methods assume the population is always sorted
    """

    def __init__(self): # TODO add settings parameter
        # read in settings
        num_dims        = settings['number_of_dimensions']
        population_size = settings['population_size']
        bounds          = settings['bounds']

        # check to make sure num_dims and number of bounds provided match
        if len(bounds) != num_dims:
            raise ValueError("Number of dimensions doesn't match number of bounds provided")

        # set instance variables
        self.settings        = settings
        # TODO move away from these next 3 vars
        self.num_dims        = num_dims
        self.population_size = population_size
        self.bounds          = bounds

        self.population      = GA.__gen_population(bounds, population_size)
        self.total_organisms = len(self.population)
        self.num_generations = 1

        self.best_x = self.population[0]

        if settings['plot']:
            try:
                self.plotutils = PlotUtils(num_dims, bounds, objective_function)
            except ValueError:
                print("Can not plot more than 2 dimensions")
                settings['plot'] = False

    @staticmethod
    def __gen_organism(id, bounds):
        # use gen_random_numbers to get a list of positions within the bounds
        return Organism(id, oa_utils.gen_random_numbers(bounds))

    @staticmethod
    def __gen_population(bounds, size):
        b = bounds
        # generate a list of organisms
        p = [GA.__gen_organism(i+1, b) for i in range(0, size)]
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
    def __selection(population, cutoff, debug=False):
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

                if debug:
                    id = organism.id
                    print("Selection: Deleting organism %d with val %f" % (id,f))

        return population # TODO test that this is still sorted

    @staticmethod
    def __get_parent_index(cdf_value, arr):
        norm_sum = 0
        for i, o in enumerate(arr):
            norm_sum += o['probability']
            if norm_sum >= cdf_value:
                return i
        return -1

    @staticmethod
    def __mate_parents(id, parent1, parent2):
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
        return (Organism(id1, pos1), Organism(id2, pos2))

    # TODO a lot going on here. probably a good idea to test the different cases
    # TODO i want to redo this. I think it will give better performance if breeding is
    # random and the best have the highest chance of getting picked for breeding
    """
        population: population
        size: size that the population should be after crossover
        NOTE: population must be sorted. crossover will return an unsorted
              array of the new population.
    """
    @staticmethod
    def __crossover(id, population, size, debug=False):
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
            child1, child2 = GA.__mate_parents(id, population[i], population[j])
            id += 2
            # append children to new_population
            new_population.extend((child1, child2))

        if debug:
            for organism in new_population:
                id = organism.id
                f  = organism.fitness
                print("Crossover: New oganism %d with val %f" % (id,f))

        return new_population

    @staticmethod
    def __mutation(population, bounds, rate, max_mutation_amount, debug=False):
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

                organism.pos = new_pos

                if debug:
                    print("Mutation: Moved organism %d to " % organism.id)
                    print(new_pos)

        return population

    def __display_state(self):
        print("implement display_state")

    def __plot_state(self):
        pts = [(organism.pos[0], organism.pos[1]) for organism in self.population]
        self.plotutils.plot(pts)

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
        # check we haven't hit a bug in the code
        if self.population_size != len(self.population):
            raise ValueError("We somehow lost track of the population. size=%d, actual=%d" \
                % (self.population_size, len(self.population)))

        population = self.population

        population = GA.__selection(population, self.settings['selection_cutoff'])

        population = GA.__crossover(self.total_organisms, population, self.settings['population_size'])
        self.total_organisms += len(population)

        population = GA.__mutation(population, self.bounds, settings['mutation_rate'], \
                      settings['max_mutation_amount'])

        self.population = GA.__sort_population(population)
        self.num_generations += 1

        if self.population[0].fitness < self.best_x.fitness:
            self.best_x = self.population[0]

        print("The best f is %f by organism %d" % (self.get_best_f(), \
                                                   self.get_best_x().id))

        if settings['step_through']:
            #self.__display_state()
            oa_utils.pause()

        if settings['plot']:
            self.__plot_state()


if __name__ == "__main__":
    ga = GA()
    print("The best f is %f" % ga.get_best_f())
    while settings['num_generations'] > ga.num_generations:
        ga.do_loop()
        time.sleep(settings['time_delay'])

    print("The best f is %f by organism %d at X =" % (ga.get_best_f(), ga.get_best_x().id))
    print(ga.get_best_x().pos)
    sys.exit()
