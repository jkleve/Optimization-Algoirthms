from numpy import sqrt, cos

def objective_function(params):
    x1 = params[0]
    x2 = params[1]
    # return (x1**2 + x2**2)
    return ( 1.0 + 1.0/4000.0*(x1**2 + x2**2) - ( cos(x1/sqrt(1.0))*cos(x2/sqrt(2.0)) ) )
