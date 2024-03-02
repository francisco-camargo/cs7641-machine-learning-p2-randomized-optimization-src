import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'queens',
        'problem_size': 4,
        },
    'rand_optimization_dict' : {
        'seed': range(2),
        'iteration_list': np.round(np.logspace(0, 2, 20)),
        'max_attempts': 1000,
        },
    'algo_dict' : {
        'algo': 'rhc',
        'restart_list': [10],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)