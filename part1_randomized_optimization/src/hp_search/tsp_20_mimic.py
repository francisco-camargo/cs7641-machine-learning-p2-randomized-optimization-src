import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'tsp',
        'problem_size': 20,
        },
    'rand_optimization_dict' : {
        'seed': range(1),
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 50,
        },
    'algo_dict' : {
        'algo': 'mimic',
        'population_sizes': [3e3],
        'keep_percent_list':[0.1, 0.3, 0.5, 0.7, 0.9],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)