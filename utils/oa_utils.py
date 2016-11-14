import random

def gen_random_numbers(bounds):
    b = bounds
    return [random.randint(b[i][0], b[i][1]) for i in range(0, len(bounds))]

import sys # check which version of python is runnint
# check if running with python3 or python2
PY3 = sys.version_info[0] == 3

def pause():
    if PY3:
        input("Waiting for Enter to be pressed ...")
    else:
        raw_input("Waiting for Enter to be pressed ...")

if __name__ == "__main__":
    bounds = [(-10,10), (-10,10)]
    print(gen_random_numbers(bounds))
