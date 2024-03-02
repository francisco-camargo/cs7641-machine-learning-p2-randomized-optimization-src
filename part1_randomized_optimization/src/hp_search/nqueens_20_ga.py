import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'queens',
        'problem_size': 20,
        },
    'rand_optimization_dict' : {
        'seed': range(3),
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 100,
        },
    'algo_dict' : {
        'algo': 'ga',
        'population_sizes':[20, 25, 30, 35, 40],
        'mutation_rates':[0.95, 0.99, 0.995],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)