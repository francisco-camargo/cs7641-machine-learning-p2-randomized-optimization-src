import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'cpeaks',
        'problem_size': 40,
        't_pct': 0.1,
        },
    'rand_optimization_dict' : {
        'seed': range(1),
        'iteration_list': np.round(np.logspace(0, 1.5, 20)),
        'max_attempts': 20,
        },
    'algo_dict' : {
        'algo': 'mimic',
        'population_sizes':[1000,2000,3000,4000],
        'keep_percent_list':[0.01, 0.1, 0.2],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)