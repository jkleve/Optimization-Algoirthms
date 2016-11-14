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
    """

    def __init__(self):
        # read in settings
        num_dims        = settings['number_of_dimensions']
        population_size = settings['population_size']
        bounds          = settings['bounds']

        # check to make sure num_dims and number of bounds provided match
        if len(bounds) != num_dims:
            raise ValueError("Number of dimensions doesn't match number of bounds provided")

        # set instance variables
        self.num_dims        = num_dims
        self.population_size = population_size
        self.bounds          = bounds
        self.population      = GA.__gen_population(bounds, population_size)
        self.total_organisms = len(self.population)
        self.num_generations = 1

        if settings['plot']:
            try:
                self.plotutils = PlotUtils(num_dims, bounds)
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
    '''
    @staticmethod
    def __selection(population):
        size    = len(population)
        max_val = population[0].fitness
        min_val = population[size-1].fitness

        # denominator in probability of surviving
        den = (max_val - min_val)
        if den == 0:
            print("Every organism has same objective function value.")

        for (i, organism) in enumerate(population):
            v = organism.fitness

            # check for division by zero
            if den == 0: prob = 0
            else: prob = float(v - min_val) / den

            if prob*settings['selection_multiplier'] > settings['selection_cutoff']:
                if settings['debug']:
                    id = organism.id
                    f = organism.fitness
                    print("Selection: Deleting organism %d with val %f" % (id,f))
                # delete the organism from the population
                del population[i]
        return population # TODO test that this is still sorted

    def __mate_organisms(self, parent1, parent2):
        pos1 = parent1.pos
        pos2 = parent2.pos
        n = len(pos1)
        split = random.randint(0, n-1)
        pos1 = pos1[0:split] + pos2[split:]
        pos2 = pos2[0:split] + pos1[split:]
        id1 = self.total_organisms + 1
        id2 = self.total_organisms + 2
        self.total_organisms += 2

        return (Organism(id1, pos1), Organism(id2, pos2))

    # TODO a lot going on here. probably a good idea to test the different cases
    # TODO i want to redo this. I think it will give better performance if breeding is
    # random and the best have the highest chance of getting picked for breeding
    def __crossover(self):
        # just do random partners for simplicity
        to_breed = list(self.population)
        pop_size = len(to_breed)
        new_population = []

        # breed each parent once
        while len(to_breed) > 1:
            p1 = to_breed[random.randint(0,pop_size-1)]
            to_breed.remove(p1)
            pop_size = len(to_breed)
            p2 = to_breed[random.randint(0,pop_size-1)]
            to_breed.remove(p2)
            pop_size = len(to_breed)
            child1, child2 = self.__mate_organisms(p1, p2)
            new_population.append(child1)
            new_population.append(child2)

        pop_size = len(self.population)

        # if we missed a parent, breed them with a rando
        if len(to_breed) > 0:
            p1 = to_breed[0]
            p2 = self.population[random.randint(0,pop_size-1)]
            child1, child2 = self.__mate_organisms(p1, p2)
            # choose only one of the childs to be used
            if random.randint(0,1) == 0: new_population.append(child1)
            else: new_population.append(child2)

        # continue getting more offspring until we reach our population size
        while len(new_population) < self.population_size:
            p1 = self.population[random.randint(0,pop_size-1)]
            p2 = self.population[random.randint(0,pop_size-1)]
            child1, child2 = self.__mate_organisms(p1, p2)
            # choose only one of the childs to be used
            if random.randint(0,1) == 0: new_population.append(child1)
            else: new_population.append(child2)

        if settings['debug']:
            for organism in new_population:
                id = organism.id
                f  = organism.fitness
                print("Crossover: New oganism %d with val %f" % (id,f))

        self.population = new_population # TODO does this need to be a new list? possible bug

    def __mutation(self):
        for organism in self.population:
            if random.random() < settings['mutation_rate']:
                new_pos = []
                for i in range(0, len(self.bounds)):
                    # take some percentage of the max mutation amount
                    x = random.uniform(0.01, 1.00)
                    delta_pos = (-1.0*log(1-x))*settings['mutation_amount']
                    # should we go positive or negative
                    if random.randint(0,1) == 1: delta_pos = -1.0*delta_pos
                    new_dim_pos = organism.pos[i] + delta_pos
                    # cap where we can go if we are beyond the bounds of the design space
                    if new_dim_pos < self.bounds[i][0]: new_dim_pos = self.bounds[i][0]
                    elif new_dim_pos > self.bounds[i][1]: new_dim_pos = self.bounds[i][1]
                    new_pos.append(new_dim_pos)

                organism.pos = new_pos
                if settings['debug']:
                    print("Mutation: Moved organism %d to " % organism.id)
                    print(new_pos)

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
        return self.population[0]

    def get_best_f(self):
        return self.population[0].fitness

    def do_loop(self):
        # check we haven't hit a bug in the code
        if self.population_size != len(self.population):
            raise ValueError("We somehow lost track of the population. size=%d, actual=%d" \
                % (self.population_size, len(self.population)))

        population = self.population

        population = GA.__selection(population)

        # TODO change crossover and mutation to accept population and
        # return an array
        self.population = population
        self.__crossover()
        self.__mutation()

        #self.population = population
        self.num_generations += 1

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
        time.sleep(0.01)

    print("The best f is %f" % ga.get_best_f())
    print(ga.get_best_x().id)
    print(ga.get_best_x().pos)
    sys.exit()
