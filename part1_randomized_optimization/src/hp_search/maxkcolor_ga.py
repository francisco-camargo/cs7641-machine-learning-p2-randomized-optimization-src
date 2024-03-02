import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'maxkcolor',
        'problem_size': 100,
        },
    'rand_optimization_dict' : {
        'seed': range(1),
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 100,
        },
    'algo_dict' : {
        'algo': 'ga',
        'population_sizes':[3, 10, 30, 100],
        'mutation_rates':[0.1, 0.4, 0.6, 0.9, 0.95, 0.99],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)