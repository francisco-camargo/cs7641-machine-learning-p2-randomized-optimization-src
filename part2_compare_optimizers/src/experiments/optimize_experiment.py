'''
Code to run explore hyperparameter space
'''

import mlrose_hiive as mlh
import numpy as np

import src.helper_functions as helpers

def run_experiment(
    algo_nickname,
    grid_search_scorer_method,
    x_train,
    y_train,
    x_test,
    y_test,
    ):
    
    # Set a name to pass to NNGSRunner
    experiment_name = 'dummy_name'
    
    # By definition of what function this is, we want to optimize and use the appropriate hyperparameters
    mode='optimize'
    
    # Get hyperparameters
    exp_out = helpers.exp_setup(algo_nickname=algo_nickname, mode=mode)
    algorithm, cv, grid_search_parameters, nn_seed_iterable, algo_seed_iterable = exp_out
    
    for nn_seed in nn_seed_iterable:
        print()
        print(f'\tnn_seed: {nn_seed}')
        for algo_seed in algo_seed_iterable:
            print(f'\t\talgo_seed: {algo_seed}')
            grid_search_parameters['seed'] = [algo_seed]
            clfRunner = mlh.NNGSRunner(
                                    x_train=x_train,
                                    y_train=y_train,
                                    x_test=x_test,
                                    y_test=y_test,
                                    experiment_name=experiment_name,
                                    algorithm=algorithm,
                                    grid_search_parameters=grid_search_parameters,
                                    grid_search_scorer_method=grid_search_scorer_method,
                                    iteration_list=np.round(np.logspace(0, 3, 20)),
                                    bias=True,
                                    early_stopping=True,
                                    clip_max=1e+10,
                                    max_attempts=10, # use default value
                                    cv=cv,
                                    generate_curves=True,
                                    seed=nn_seed,
                                    n_jobs=-1
                                    )
            
            run_stats_df, curves_df, cv_results_df, grid_search_cv = clfRunner.run() # GridSearchCV instance returned
            # for key, val in cv_results_df.items():
            #     print(key, val)
            try:
                print(cv_results_df[['mean_fit_time', 'param_learning_rate_init', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']])
            except KeyError:
                print(cv_results_df[['mean_fit_time', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']])
            print(f'\t\t{grid_search_cv.best_params_}')
            print(f'\t\tBest score: {grid_search_cv.best_score_}')
            
    print()