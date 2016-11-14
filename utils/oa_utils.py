import random

def gen_random_numbers(bounds):
    b = bounds
    return [random.randint(b[i][0], b[i][1]) for i in range(0, len(bounds))]

if __name__ == "__main__":
    bounds = [(-10,10), (-10,10)]
    print(gen_random_numbers(bounds))
