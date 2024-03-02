import mlrose_hiive
import numpy as np
import src.sa_setup as sa_setup

# Configure the temperature values to use
temp_type   = [mlrose_hiive.ExpDecay]
init_temp   = [1e1, 1e2, 1e3, 1e4]
decay       = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]
min_temp    = [1e-5]

# Generate temperature_list in needed format
temperature_list = sa_setup.sa_setup(temp_type=temp_type, init_temp=init_temp, decay=decay, min_temp=min_temp)

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'tsp',
        'problem_size': 10,
        },
    'rand_optimization_dict' : {
        'seed': range(10),
        'iteration_list': np.round(np.logspace(0, 3, 20)),
        'max_attempts': 50,
        },
    'algo_dict' : {
        'algo': 'sa',
        'temperature_list' : temperature_list
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)