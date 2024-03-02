import numpy as np
import mlrose_hiive
import src.sa_setup as sa_setup

# Configure the temperature values to use
temp_type   = [mlrose_hiive.ExpDecay]
init_temp   = [5e3]
decay       = [0.5]
min_temp    = [1e-4]

# Generate temperature_list in needed format
temperature_list = sa_setup.sa_setup(temp_type=temp_type, init_temp=init_temp, decay=decay, min_temp=min_temp)

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'flipflop',
        'problem_size': np.round(np.logspace(1, 2, 6)),
        'fixed_problem_size': 30 # use whatever value was used to determine optimal hyperparameters
        },
    'rand_optimization_dict' : {
        'seed': range(5),
        'iteration_list': np.round(np.logspace(0, 3, 20)),
        'max_attempts': 100,
        },
    'algo_dict' : {
        'rhc':{
            'restart_list': [10],
            },
        'sa':{
            'temperature_list': temperature_list
            },
        'ga':{
            'population_sizes':[100],
            'mutation_rates':[0.2],
            },
        'mimic':{
            'population_sizes':[1000],
            'keep_percent_list':[0.05],
            }
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)