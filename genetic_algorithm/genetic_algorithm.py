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
            self.population.append(Organism(i+1, self.num_dims, pos))

    def get_rand_pos(self):
        b = self.bounds
        return [random.randint(b[i][0], b[i][1]) for i in range(0, self.num_dims)]

if __name__ == "__main__":
    ga = GA()
