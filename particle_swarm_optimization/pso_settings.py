settings = {
    'population_size': 500, # number of organisms
    'number_of_dimensions': 2,  # if you change the dimensions, you need to change bounds and change plot to false
    'bounds': [ # this must have 1 pair per dimension
        (-10,10),
        (-10,10)
    ],
    'num_iterations': 100,
    'time_delay': 0.0,
    ##### Leave these between 0 and 1 #####

    'cp': 0.2,          # Weight that has the particles tend to go to local known minimum
    'cg': 0.8,          # Weight that has the particles tend to go to global minimum
    'weight': 0.5,      # this is only used when velocity_type is inertia... haven't really seen much change though when altering this
    'velocity_type': 'inertia', # normal, inertia, or constriction
    'step_through': False, # whether it should pause after each iteration
    'plot': True, # whether we should plot each iteration
    'print_actions': False, # whether we should print when a method does an action
    'print_iterations': False, # whether we should print after each iteration
    'time': False, # whether we should output timing information

    'time_as_float': True
}
