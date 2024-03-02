import mlrose_hiive as mlh
import itertools

def sa_setup(temp_type, init_temp, decay, min_temp):
    # Compile the temp info into usable form
    temp_prep = list(itertools.product(*[temp_type, init_temp, decay, min_temp]))
    temperature_list = []
    for item in temp_prep:
        temp_type, init_temp, decay, min_temp = item
        temperature_list += [temp_type(init_temp, decay, min_temp)]
    return temperature_list

# both seeds affect score by +/-0.05 (5%)

# Configure the temperature values to use
temp_type   = [mlh.ExpDecay]
init_temp   = [0.01]
decay       = [0.1]
min_temp    = [1e-10]

# Generate temperature_list in needed format
temperature_list = sa_setup(temp_type=temp_type, init_temp=init_temp, decay=decay, min_temp=min_temp)

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'monks',
        'algo': 'sa',
        'nn_seed': range(1,2), # confirmed to affect results
        'algo_seed': range(1,2), # confirmed to affect results
        'cv':3,
        },
    'nn_dict': {
        'learning_rate_init': [0.4],
        'activation': [mlh.neural.activation.relu],
        'is_classifier': [True],
        'hidden_layer_sizes': [[4]],
        },
    'algo_dict' : {
        'max_attempts': [100],
        'max_iters': [100],
        'schedule': temperature_list,
        },
    }



# Configure the temperature values to use
temp_type   = [mlh.ExpDecay]
init_temp   = [0.01]
decay       = [0.1]
min_temp    = [1e-10]

# Generate temperature_list in needed format
temperature_list = sa_setup(temp_type=temp_type, init_temp=init_temp, decay=decay, min_temp=min_temp)

# score of about 0.9
best_dict = {
    'problem_dict' : {
        'problem': 'monks',
        'algo': 'sa',
        'nn_seed': range(1,4),
        'algo_seed': range(1,4),
        'cv':5,
        },
    'nn_dict': {
        'learning_rate_init': [0.4],
        'activation': [mlh.neural.activation.relu],
        'is_classifier': [True],
        'hidden_layer_sizes': [[4]],
        },
    'algo_dict' : {
        'max_attempts': [100],
        'max_iters': [1000],
        'schedule': temperature_list,
        },
}

def get_exp(mode='best'):
    if mode == 'optimize':
        return experiment_dict
    if mode == 'best':
        return best_dict
    if mode == 'fitness':
        return best_dict
    if mode == 'learning':
        return best_dict
    
if __name__ == '__main__':
    print(experiment_dict)