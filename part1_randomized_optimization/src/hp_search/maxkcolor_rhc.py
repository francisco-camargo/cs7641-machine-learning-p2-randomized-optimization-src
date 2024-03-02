import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'maxkcolor',
        'problem_size': 50,
        },
    'rand_optimization_dict' : {
        'seed': range(1),
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 100,
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