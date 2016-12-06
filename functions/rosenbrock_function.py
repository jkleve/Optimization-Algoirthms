def objective_function(params):
    x1 = params[0]
    x2 = params[1]

    return ( 100*(x2 - x1*x1)**2 + (x1 - 1)**2 )
