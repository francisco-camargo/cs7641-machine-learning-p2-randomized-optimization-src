import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'flipflop',
        'problem_size': 4,
        },
    'rand_optimization_dict' : {
        'seed': range(1),
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 1000,
        },
    'algo_dict' : {
        'algo': 'mimic',
        'population_sizes':[10,30,100,300],
        'keep_percent_list':[0.1,0.3,0.7,0.9],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)