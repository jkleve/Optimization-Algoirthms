import matplotlib.pyplot as plt # plotting
import random # randint
import sys # to exit
import time # delay

from ga_settings import settings
from ga_objective_function import objective_function

class Organism:
    id = 0
    num_dims = 0
    pos = []
    f = 0
    func = None

    def __init__(self, id, num_dims, pos, func=objective_function):
        self.id = id
        self.num_dims = num_dims
        self.pos = pos
        self.func = func
        self.f = self.fitness()

    def fitness(self):
        return self.func(self.pos)

class GA:
    num_dims = 0
    population_size = 0
    total_organisms = 0
    population = []
    bounds = []

    def __init__(self):
        self.num_dims        = settings['number_of_dimensions']
        self.population_size = settings['population_size']
        self.bounds          = settings['bounds']

        # check to make sure num_dims and number of bounds provided match
        if len(self.bounds) != self.num_dims:
            raise ValueError("Number of dimensions doesn't match number of bounds provided")

        self.init_population()

    def init_population(self):
        for i in range(0, self.population_size):
            pos = self.get_rand_pos()
            self.population.append(Organism(i+1, self.num_dims, pos, objective_function))
            self.total_organisms += 1

    def get_rand_pos(self):
        b = self.bounds
        return [random.randint(b[i][0], b[i][1]) for i in range(0, self.num_dims)]

    ###########################
    ###  GA steps and loop  ###
    ###########################
    def selection(self):
        population_values = [getattr(organism, 'f') for organism in self.population]
        max_val = max(population_values)
        min_val = min(population_values)
        for organism in self.population:
            v = getattr(organism, 'f')
            prob = float(v - min_val) / (max_val - min_val)
            if prob*settings['selection_multiplier'] > settings['selection_cutoff']:
                self.population.remove(organism) # TODO this may be too slow but easy to read

    def new_organism(self, parent1, parent2):
        pos1 = getattr(parent1, 'pos')
        pos2 = getattr(parent2, 'pos')
        n = len(pos1)
        split = random.randint(0, n-1)
        pos1 = pos1[0:split] + pos2[split:]
        pos2 = pos2[0:split] + pos1[split:]
        id1 = self.total_organisms + 1
        id2 = self.total_organisms + 2
        self.total_organisms += 2
        return (Organism(id1, self.num_dims, pos1), Organism(id2, self.num_dims, pos2))

    # TODO a lot going on here. probably a good idea to test the different cases
    def crossover(self):
        # just do random partners for simplicity
        to_breed = list(self.population)
        pop_size = len(to_breed)
        new_population = []

        # bread each parent once
        while len(to_breed) > 1:
            p1 = to_breed[random.randint(0,pop_size-1)]
            to_breed.remove(p1)
            pop_size = len(to_breed)
            p2 = to_breed[random.randint(0,pop_size-1)]
            to_breed.remove(p2)
            pop_size = len(to_breed)
            child1, child2 = self.new_organism(p1, p2)
            new_population.append(child1)
            new_population.append(child2)

        pop_size = len(self.population)

        # if we missed a parent, bread them with a rando
        if len(to_breed) > 0:
            p1 = to_breed[0]
            p2 = self.population[random.randint(0,pop_size-1)]
            child1, child2 = self.new_organism(p1, p2)
            # choose only one of the childs to be used
            if random.randint(0,1) == 0: new_population.append(child1)
            else: new_population.append(child2)

        # continue getting more offspring until we reach our population size
        while len(new_population) < self.population_size:
            p1 = self.population[random.randint(0,pop_size-1)]
            p2 = self.population[random.randint(0,pop_size-1)]
            child1, child2 = self.new_organism(p1, p2)
            # choose only one of the childs to be used
            if random.randint(0,1) == 0: new_population.append(child1)
            else: new_population.append(child2)

        self.population = new_population # TODO does this need to be a new list? possible bug

    def mutation(self):
        print("implement")

    def next_generation(self):
        # check we haven't hit a bug in the code
        if self.population_size != len(self.population):
            raise ValueError("We somehow lost track of the population. size=%d, actual=%d" \
                % (self.population_size, len(self.population)))

        self.selection()
        self.crossover()
        self.mutation()

        if settings['step_through'] is True:
            self.display_state()
            input("Paused. Hit Enter to continue")

        if settings['plot'] is True:
            self.plot_state()
            #input("Paused. Hit Enter to continue")

    def display_state(self):
        print("implement")

    def plot_state(self):
        if self.num_dims > 2:
            print("Can not plot more than 2 dimensions")
            settings['plot'] = False

        x = [getattr(organism, 'pos')[0] for organism in self.population]
        y = [getattr(organism, 'pos')[1] for organism in self.population]
        plt.plot(x,y, 'ro')
        plt.show()

if __name__ == "__main__":
    ga = GA()
    while True:
        ga.next_generation()
        time.sleep(0.1)

    sys.exit()
