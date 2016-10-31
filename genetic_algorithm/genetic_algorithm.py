import random
import sys

from ga_settings import settings
from ga_objective_function import objective_function as func

class Organism:
    id = 0
    num_dims = 0
    pos = []
    f = 0
    func = None

    def __init__(self, id, num_dims, pos, func):
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
            self.population.append(Organism(i+1, self.num_dims, pos, func))

    def get_rand_pos(self):
        b = self.bounds
        return [random.randint(b[i][0], b[i][1]) for i in range(0, self.num_dims)]

    def next_generation(self):
        print("to implement ...")

if __name__ == "__main__":
    ga = GA()
    sys.exit()

    params = []
    params.append(2)
    params.append(3)
    print(func(params))
    sys.exit()
