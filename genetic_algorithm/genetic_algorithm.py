import matplotlib.pyplot as plt # plotting
import random # randint
import sys # to exit
import time # delay
from math import log

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
    fig = None
    best_f = float("inf")
    best_organism = None
    num_generations = 0

    def __init__(self):
        self.num_dims        = settings['number_of_dimensions']
        self.population_size = settings['population_size']
        self.bounds          = settings['bounds']

        # check to make sure num_dims and number of bounds provided match
        if len(self.bounds) != self.num_dims:
            raise ValueError("Number of dimensions doesn't match number of bounds provided")

        self.init_population()
        self.num_generations += 1
        self.best_f = min([organism.f for organism in self.population])
        self.best_organism = self.get_best_organism()

        if settings['plot'] is True:
            if self.num_dims > 2:
                print("Can not plot more than 2 dimensions")
                settings['plot'] = False
            else:
                self.fig, self.ax = plt.subplots()
                self.line, = self.ax.plot([], [], 'ro')
                self.ax.grid()
                xlim_l = settings['bounds'][0][0]
                xlim_u = settings['bounds'][0][1]
                ylim_l = settings['bounds'][1][0]
                ylim_u = settings['bounds'][1][1]
                self.ax.set_xlim(xlim_l, xlim_u)
                self.ax.set_ylim(ylim_l, ylim_u)
                #plt.show()

    def init_population(self):
        for i in range(0, self.population_size):
            pos = self.get_rand_pos()
            self.population.append(Organism(i+1, self.num_dims, pos, objective_function))
            self.total_organisms += 1

    def get_rand_pos(self):
        b = self.bounds
        return [random.randint(b[i][0], b[i][1]) for i in range(0, self.num_dims)]

    def get_best_organism(self):
        best = None
        for organism in self.population:
            if best == None or best.f < organism.f:
                best = organism
        return best

    ###########################
    ###  GA steps and loop  ###
    ###########################
    def selection(self):
        population_values = [getattr(organism, 'f') for organism in self.population]
        max_val = max(population_values)
        min_val = min(population_values)

        den = (max_val - min_val)
        if den == 0:
            print("Every organism has same objective function value.")

        for organism in self.population:
            v = getattr(organism, 'f')

            # check for division by zero
            if den == 0: prob = 0
            else: prob = float(v - min_val) / den

            if prob*settings['selection_multiplier'] > settings['selection_cutoff']:
                if settings['debug']:
                    id = getattr(organism, 'id')
                    f = getattr(organism, 'f')
                    print("Selection: Removing organism %d with val %f" % (id,f))
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

        if settings['debug']:
            for organism in new_population:
                id = getattr(organism, 'id')
                f = getattr(organism, 'f')
                print("Crossover: New oganism %d with val %f" % (id,f))

        self.population = new_population # TODO does this need to be a new list? possible bug

    def mutation(self):
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

    def next_generation(self):
        # check we haven't hit a bug in the code
        if self.population_size != len(self.population):
            raise ValueError("We somehow lost track of the population. size=%d, actual=%d" \
                % (self.population_size, len(self.population)))

        self.selection()
        self.crossover()
        self.mutation()
        self.num_generations += 1
        self.best_organism = self.get_best_organism()
        self.best_f = self.best_organism.f

        print("The best f is %f by organism %d" % (self.best_f, self.best_organism.id))

        if settings['step_through'] is True:
            self.display_state()
            input("Paused. Hit Enter to continue")

        if settings['plot'] is True:
            self.plot_state()
            #input("Paused. Hit Enter to continue")

    def display_state(self):
        print("implement display_state")

    def plot_state(self):
        x = [getattr(organism, 'pos')[0] for organism in self.population]
        y = [getattr(organism, 'pos')[1] for organism in self.population]
        self.line.set_xdata(x)
        self.line.set_ydata(y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    plt.ion()
    ga = GA()
    print("The best f is %f" % ga.best_f)
    while settings['num_generations'] > ga.num_generations:
        ga.next_generation()
        time.sleep(0.1)

    print("The best f is %f" % ga.best_f)
    print(ga.best_organism.id)
    print(ga.best_organism.pos)
    sys.exit()
