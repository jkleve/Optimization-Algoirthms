from numpy import cos, exp, pi

def objective_function(params):
    x1 = params[0]
    x2 = params[1]

    return ( -1.*cos(x1)*cos(x2)*exp(-1.*((x1-pi)**2 + (x2-pi)**2)) )
