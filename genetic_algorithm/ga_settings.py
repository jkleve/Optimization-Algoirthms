settings = {
    'population_size': 100, # number of organisms
    'number_of_dimensions': 2,
    'bounds': [ # this must have 1 pair per dimension
        (-10,10),
        (-10,10)
    ],
    'selection_cutoff': 0.8,
    'mutation_rate': 0.1, # how often a mutation happens (between 0 and 1)
    'max_mutation_amount': 0.4, # how large of a mutation is allowed (between 0 and 1)
    'num_generations': 100,
    'time_delay': 0.0,
    'step_through': False, # whether it should pause after each iteration
    'plot': False, # whether we should plot each iteration
    'debug': False, # whether we should print each iteration
    'time': True # whether we should output timing information
}
