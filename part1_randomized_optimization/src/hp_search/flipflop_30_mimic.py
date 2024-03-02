import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'flipflop',
        'problem_size': 30,
        },
    'rand_optimization_dict' : {
        'seed': range(2),
        'iteration_list': np.round(np.logspace(0, 1.5, 20)),
        'max_attempts': 100,
        },
    'algo_dict' : {
        'algo': 'mimic',
        'population_sizes':[1000,1500,2000],
        'keep_percent_list':[0.01,0.05,0.1],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)