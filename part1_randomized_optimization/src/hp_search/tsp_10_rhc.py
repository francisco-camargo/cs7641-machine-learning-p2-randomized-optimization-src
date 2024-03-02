import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'tsp',
        'problem_size': 10,
        },
    'rand_optimization_dict' : {
        'seed': range(3),
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 50,
        },
    'algo_dict' : {
        'algo': 'rhc',
        'restart_list': [1,10,100]
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)