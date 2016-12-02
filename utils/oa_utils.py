import random

def optimize_settings(s):
    s['time_delay'] = 0.0
    s['step_through'] = False
    s['plot'] = False
    s['print_actions'] = False
    s['print_iterations'] = False
    s['time'] = False
    return s

def gen_random_numbers(bounds):
    b = bounds
    return [random.uniform(b[i][0], b[i][1]) for i in range(0, len(bounds))]

import sys # check which version of python is runnint
# check if running with python3 or python2
PY3 = sys.version_info[0] == 3

def get_even_spread(bounds, num_parts):
    print("implement")

def pause():
    if PY3:
        input("Waiting for Enter to be pressed ...")
    else:
        raw_input("Waiting for Enter to be pressed ...")

def count_num_lines(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
        return (i + 1)

def count_num_vars(filename):
    with open(filename) as f:
        for row in f:
            return ( len(row.split(',')) - 1 )

def write_xy_data(X, y, filename):
    with open(filename, 'w') as f:
        for i, row in enumerate(X):
            row_str = ""
            for x in row:
                row_str += str(x) + ','
            row_str += str(y[i][0]) + '\n'
            f.write(row_str)

def read_xy_data(filename):
    import numpy as np

    with open(filename, 'r') as f:
        n = count_num_lines(filename)
        m = count_num_vars(filename)

        x = np.zeros(shape=(n,m))
        y = np.zeros(shape=(n,1))

        for i, row_str in enumerate(f):
            row = row_str.split(',')
            y[i] = [-1]
            for j, var in enumerate(row[0:-1]):
                x[i,j] = var

        return (x, y)

if __name__ == "__main__":
    #bounds = [(-10,10), (-10,10)]
    #print(gen_random_numbers(bounds))

    import numpy as np
    import sys

    #X = np.zeros(shape=(10,4))
    #Y = np.ones(shape=(10,1))

    #write_xy_data(X, Y, 'data.dat')

    x, y = read_xy_data('data.dat')

    print(x)
    print(y)
