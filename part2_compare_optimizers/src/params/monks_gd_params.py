import mlrose_hiive as mlh
import numpy as np

# Define grid search for optimization experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'monks',
        'algo': 'gd',
        'nn_seed': range(1,2), # doesn't seem to do anything for GD
        'algo_seed': range(1,2),
        'cv':3,
        },
    'nn_dict': {
        'learning_rate_init': [0.01],
        'activation': [mlh.neural.activation.relu],
        'is_classifier': [True],
        'hidden_layer_sizes': [[4]],
        },
    'algo_dict' : {
        'max_attempts': [1e3],
        'max_iters': [10,30,100,300],
        },
    }


# define 'best' parameters
best_dict = {
    'problem_dict' : {
        'problem': 'monks',
        'algo': 'gd',
        'nn_seed': range(1,2), # Don't change; doesn't seem to do anything for GD
        'algo_seed': range(1,10),
        'cv':5,
        },
    'nn_dict': {
        'learning_rate_init': [0.01],
        'activation': [mlh.neural.activation.relu],
        'is_classifier': [True],
        'hidden_layer_sizes': [[4]],
        },
    'algo_dict' : {
        'max_attempts': [1e3],
        'max_iters': [100],
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