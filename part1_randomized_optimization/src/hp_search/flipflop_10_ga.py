import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'flipflop',
        'problem_size': 10,
        },
    'rand_optimization_dict' : {
        'seed': range(3),
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 100,
        },
    'algo_dict' : {
        'algo': 'ga',
        'population_sizes':[3, 5, 10],
        'mutation_rates':[0.1, 0.2, 0.5],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)