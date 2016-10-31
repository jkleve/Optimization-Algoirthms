import random

from ga_settings import settings

class Organism:
    id = 0
    f = 0
    num_dims = 0
    pos = []

    def __init__(self, id, num_dims, pos):
        self.id = id
        self.num_dims = num_dims
        self.pos = pos
        print(id)
        print(pos)
        print(num_dims)
        print('\n')

class GA:
    num_dims = 0
    population_size = 0
    population = []
    bounds = []

    def __init__(self):
        self.num_dims = settings['number_of_dimensions']
        self.population_size = settings['population_size']
        self.bounds = settings['bounds']
        self.init_population()

    def init_population(self):
        for i in range(0, self.population_size):
            pos = self.get_rand_pos()
            self.population.append(Organism(i+1, self.num_dims, pos))

    def get_rand_pos(self):
        pos = []
        for i in range(0, self.num_dims):
            lower = self.bounds[i][0]
            upper = self.bounds[i][1]
            pos.append(random.randint(lower, upper))
        return pos

if __name__ == "__main__":
    ga = GA()
