# Optimization-Algorithms

### Linux commands for noobs
* pwd - shows you what directory you are in (working directory)
* ls or ll - shows you files in the current directory
* cd <directory> - changes to that directory (.. is the parent directory)
* python <file> - run the python file


#### Genetic Algorithm
To run, change directories with `cd genetic_algorithm` then run the program with `python genetic_algorithm.py`
* ga_objective_function.py
    - contains the function being minimized for the algorithm
* ga_settings.py
    - includes settings that will be used when running the algorithm
* genetic_algorithm.py
    - algorithm to run: "python genetic_algorithm.py"

| Setting | What it does |
| ------- | ------------ |
| population_size | # of organism to have each iteration |
| number_of_dimensions | # number of dimensions there are in the design space |
| bounds | Bounds on the design variables. You must have a list of 2 numbers for each dimension |
| selection_multiplier | A multiplier applied to each organism after it's objective function has been evaluated |
| selection_cutoff | A cutoff value that determines which organisms to kill off before crossover |
| mutation_rate | How often to cause a mutation of a child |
| mutation_amount | The largest percentage of a dimension allowed to mutate. 1.0 would mean a mutation could jump the child from one side of the bounded space to the other side. |
| num_generations | # of generations to create before exiting |
| step_through | Not implemented |
| plot | True or False on whether we should plot each iteration/generation. Can only plot when number_of_dimensions is set to 2 |
| debug | True or False on whether we should print out each iteration |
