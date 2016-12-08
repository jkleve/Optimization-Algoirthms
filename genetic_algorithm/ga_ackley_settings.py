settings = {
    'num_iterations': 200,
    'population_size': 100, # number of organisms
    'stopping_criteria': 0.001,
    'num_iter_stop_criteria': 50,
    'number_of_dimensions': 2,
    'bounds': [ # this must have 1 pair per dimension
        (-40,40),
        (-40,40)
    ],

    'selection_cutoff': 0.3,
    'mutation_rate': 0.9, # how often a mutation happens (between 0 and 1)
    'max_mutation_amount': 1.0, # how large of a mutation is allowed (between 0 and 1)

    'time_delay': 0.0,
    'step_through': False, # whether it should pause after each iteration
    'plot': False, # whether we should plot each iteration
    'print_actions': False, # whether we should print when a method does an action
    'print_iterations': False, # whether we should print after each iteration
    'time': False, # whether we should output timing information

    'time_as_float': True
}