from math import sqrt, cos

def objective_function(params):
    x = params[0]
    y = params[1]
    return ( 1.0 + 1.0/4000.0*(x**2 + y**2) - ( cos(x/sqrt(1.0))*cos(y/sqrt(2.0)) ) )
