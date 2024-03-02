'''
Helper functions used across all the experiments
'''

import mlrose_hiive as mlh

def exp_setup(algo_nickname, mode):

    # Fetch the dictionaries that hold the hyperparameters of interest
    # MONKs
    if algo_nickname == 'gd':
        import src.params.monks_gd_params as experiment_module
    elif algo_nickname == 'rhc':
        import src.params.monks_rhc_params as experiment_module
    elif algo_nickname == 'sa':
        import src.params.monks_sa_params as experiment_module
    elif algo_nickname == 'ga':
        import src.params.monks_ga_params as experiment_module
    experiment_dict = experiment_module.get_exp(mode=mode)
    problem_dict    = experiment_dict['problem_dict']

    # set cross-validation variable
    cv = problem_dict['cv']
    
    # Pick optimization algorithm
    algo_nickname = problem_dict['algo']
    if   algo_nickname == 'gd':  algorithm = mlh.algorithms.gradient_descent
    elif algo_nickname == 'rhc': algorithm = mlh.algorithms.random_hill_climb
    elif algo_nickname == 'sa':  algorithm = mlh.algorithms.simulated_annealing
    elif algo_nickname == 'ga':  algorithm = mlh.algorithms.genetic_alg
    
    # Set up gridsearch parameters
    grid_search_parameters = {
        **experiment_dict['nn_dict'],
        **experiment_dict['algo_dict'],
    }
    
    # Rename iterable variables used to set seeds
    nn_seed_iterable    = problem_dict['nn_seed']
    algo_seed_iterable  = problem_dict['algo_seed']
    
    return algorithm, cv, grid_search_parameters, nn_seed_iterable, algo_seed_iterable