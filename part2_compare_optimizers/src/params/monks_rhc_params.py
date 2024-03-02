import mlrose_hiive as mlh

# very sensitive to seed

# Define the experiment
experiment_dict = {
    'problem_dict' : {
        'problem': 'monks',
        'algo': 'rhc',
        'nn_seed': range(1,2),
        'algo_seed': range(1,2), # confirmed to affect results
        'cv':3,
        },
    'nn_dict': {
        'learning_rate_init': [0.001,0.01,0.1,0.9],
        'activation': [mlh.neural.activation.relu],
        'is_classifier': [True],
        'hidden_layer_sizes': [[4]],
        },
    'algo_dict' : {
        'max_attempts': [100],
        'max_iters': [1000],
        'restarts': [3],
        },
    }


# score of about [75, 85]
best_dict = {
    'problem_dict' : {
        'problem': 'monks',
        'algo': 'rhc',
        'nn_seed': range(1,4),
        'algo_seed': range(1,4), # confirmed to affect results
        'cv':5,
        },
    'nn_dict': {
        'learning_rate_init': [0.9],
        'activation': [mlh.neural.activation.relu],
        'is_classifier': [True],
        'hidden_layer_sizes': [[4]],
        },
    'algo_dict' : {
        'max_attempts': [100],
        'max_iters': [1000],
        'restarts': [0],
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