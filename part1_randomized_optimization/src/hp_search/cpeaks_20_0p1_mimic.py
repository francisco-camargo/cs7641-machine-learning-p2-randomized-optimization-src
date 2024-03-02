import numpy as np

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'cpeaks',
        'problem_size': 20,
        't_pct': 0.1,
        },
    'rand_optimization_dict' : {
        'seed': range(3),
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 50,
        },
    'algo_dict' : {
        'algo': 'mimic',
        'population_sizes':[1000,2000],
        'keep_percent_list':[0.001,0.003,0.01],
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)