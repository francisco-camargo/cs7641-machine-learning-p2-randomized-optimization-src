import mlrose_hiive
import numpy as np
import src.sa_setup as sa_setup

# Configure the temperature values to use
temp_type   = [mlrose_hiive.GeomDecay, mlrose_hiive.ArithDecay, mlrose_hiive.ExpDecay]
init_temp   = [0.01, 0.1, 1]
decay       = [0.1, 0.5, 0.9, 0.95, 0.99]
min_temp    = [1e-4]

# Generate temperature_list in needed format
temperature_list = sa_setup.sa_setup(temp_type=temp_type, init_temp=init_temp, decay=decay, min_temp=min_temp)

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'queens',
        'problem_size': 6,
        },
    'rand_optimization_dict' : {
        'seed': range(5),
        # 'iteration_list': 2 ** np.arange(10),
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 100,
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