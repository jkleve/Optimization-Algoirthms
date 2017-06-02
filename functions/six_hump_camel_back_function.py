import math

def objective_function(params):
    if type(params[0]) is float:
        x1 = params[0]
        x2 = params[1]
        if x1 < 0:
            x1 = -1.*x1
        if x2 < 0:
            x2 = -1.*x2
    else:
        x1 = [abs(v) for v in params[0]]
        x2 = [abs(v) for v in params[1]]

    print x1
    print x2
    return ( (4 - 2.1*x1*x1 + math.pow(x1, 4./3.)*x1*x1 + x1*x2 + (-4 + 4*x2*x2)*x2*x2) )
