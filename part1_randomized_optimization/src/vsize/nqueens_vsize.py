import numpy as np
import mlrose_hiive
import src.sa_setup as sa_setup

# Configure the temperature values to use
temp_type   = [mlrose_hiive.ExpDecay]
init_temp   = [100]
decay       = [0.3]
min_temp    = [1e-4]

# Generate temperature_list in needed format
temperature_list = sa_setup.sa_setup(temp_type=temp_type, init_temp=init_temp, decay=decay, min_temp=min_temp)

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'queens',
        'problem_size': np.round(np.logspace(0.6, 1.7, 6)),
        'fixed_problem_size': 20 # use whatever value was used to determine optimal hyperparameters
        },
    'rand_optimization_dict' : {
        'seed': range(5),
        'iteration_list': np.round(np.logspace(0, np.log10(3000), 20)),
        'max_attempts': 100,
        },
    'algo_dict' : {
        'rhc':{
            'restart_list': [20],
            },
        'sa':{
            'temperature_list': temperature_list
            },
        'ga':{
            'population_sizes':[30],
            'mutation_rates':[0.99],
            },
        # 'mimic':{
        #     'population_sizes':[900],
        #     'keep_percent_list':[0.3],
        #     }
        },
    }

def get_exp():
    return experiment_dict
    
if __name__ == '__main__':
    print(experiment_dict)