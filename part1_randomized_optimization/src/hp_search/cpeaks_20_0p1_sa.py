import mlrose_hiive
import numpy as np
import src.sa_setup as sa_setup

# Configure the temperature values to use
temp_type   = [mlrose_hiive.ExpDecay]
init_temp   = [1e-2,1e-1,1e0,1e1]
decay       = [1e-2,1e-1,1e0,1e1]
min_temp    = [1e-10]

# Generate temperature_list in needed format
temperature_list = sa_setup.sa_setup(temp_type=temp_type, init_temp=init_temp, decay=decay, min_temp=min_temp)

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'cpeaks',
        'problem_size': 40,
        't_pct': 0.1,
        },
    'rand_optimization_dict' : {
        'seed': range(3),
        'iteration_list': np.round(np.logspace(0, 4, 20)),
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