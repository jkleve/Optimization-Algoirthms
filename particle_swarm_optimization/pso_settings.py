settings = {
    'population_size': 100, # number of organisms
    'number_of_dimensions': 2,
    'bounds': [ # this must have 1 pair per dimension
        (-10,10),
        (-10,10)
    ],
    'num_iterations': 100,
    'time_delay': 0.0,
    'cp': 0.5,
    'cg': 0.75,
    'weight': 0.5,      #this is only used when velocity_type is inertia
    'velocity_type': 'inertia' # inertia or constriction
    'step_through': False, # whether it should pause after each iteration
    'plot': False, # whether we should plot each iteration
    'print_actions': False, # whether we should print when a method does an action
    'print_iterations': False, # whether we should print after each iteration
    'time': False # whether we should output timing information
}
