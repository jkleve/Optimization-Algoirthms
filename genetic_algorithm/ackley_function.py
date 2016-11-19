from numpy import sqrt, cos, exp, pi

# Ackley Function
def ackley_function(params):
    x1 = params[0]
    x2 = params[1]

    a = 20
    b = 0.2
    c = 2*pi
    d = len(params) # number of dimensions
    
    return ( -1.0*a*exp(-1.0*b*sqrt((1.0/d)*(x1**2 + x2**2))) - exp((1.0/d)*(cos(c*x1) + cos(c*x2))) + a + exp(1) )
