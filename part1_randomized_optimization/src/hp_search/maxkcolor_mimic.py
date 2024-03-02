import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'maxkcolor',
        'problem_size': 50,
        },
    'rand_optimization_dict' : {
        'seed': range(3),
        'iteration_list': np.round(np.logspace(0, 1.5, 20)),
        'max_attempts': 100,
        },
    'algo_dict' : {
        'algo': 'mimic',
        'population_sizes': [300,900,1200],
        'keep_percent_list':[0.01,0.03,0.1],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)