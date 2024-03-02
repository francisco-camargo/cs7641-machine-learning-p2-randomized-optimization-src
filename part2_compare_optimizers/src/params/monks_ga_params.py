import mlrose_hiive as mlh

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'monks',
        'algo': 'ga',
        'nn_seed': range(1,2), # ga not too sensitive to seed
        'algo_seed': range(1,2), # does not do anything for GA...
        'cv':3,
        },
    'nn_dict': {
        'activation': [mlh.neural.activation.relu],
        'is_classifier': [True],
        'hidden_layer_sizes': [[4]],
        },
    'algo_dict' : {
        'pop_size':[60], # more pop is better but it's expensive
        'pop_breed_percent':[0.75],
        'mutation_prob':[0.1],
        'max_attempts': [100],
        'max_iters': [1000],
        },
    }


# score of about 71
best_dict = {
    'problem_dict' : {
        'problem': 'monks',
        'algo': 'ga',
        'nn_seed': range(1,4), # ga not too sensitive to seed
        'algo_seed': range(1,4), # does not do anything for GA...
        'cv':5,
        },
    'nn_dict': {
        'activation': [mlh.neural.activation.relu],
        'is_classifier': [True],
        'hidden_layer_sizes': [[4]],
        },
    'algo_dict' : {
        'pop_size':[5], # more pop is better but it's expensive
        'pop_breed_percent':[0.75],
        'mutation_prob':[0.1],
        'max_attempts': [100],
        'max_iters': [1000],
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
    