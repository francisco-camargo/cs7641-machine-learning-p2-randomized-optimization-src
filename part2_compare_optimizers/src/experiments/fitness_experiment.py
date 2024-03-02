'''
Code to run train NN with best known hyperparameters and then plot Fitness vs Iteration
'''

import mlrose_hiive as mlh
import numpy as np
import pandas as pd

import src.helper_functions as helpers

def run_experiment(
    algorithm_list,
    grid_search_scorer_method,
    x_train,
    y_train,
    x_test,
    y_test,
    ):
    
    # Set a name to pass to NNGSRunner
    experiment_name = 'dummy_name'
    
    # By definition of what function this is, we want to optimize and use the appropriate hyperparameters
    mode='best'
    
    # Select what columns to keep track of
    keep_col = [
        'Iteration',
        'Fitness',
        'FEvals',
        'Time',
        'algorithm',
    ]
    
    # Initialize seed counter
    num_seeds = 0
    
    ########################################
    # Now run the experiments
    for algo_nickname in algorithm_list: # Pick optimization algorithm
        print()
        print(f'Use algorithm: {algo_nickname}')
        
        # Get hyperparameters
        exp_out = helpers.exp_setup(algo_nickname=algo_nickname, mode=mode)
        algorithm, cv, grid_search_parameters, nn_seed_iterable, algo_seed_iterable = exp_out
    
        for nn_seed in nn_seed_iterable:
            print()
            print(f'\tnn_seed: {nn_seed}')
            for algo_seed in algo_seed_iterable:
                print(f'\t\talgo_seed: {algo_seed}')
                grid_search_parameters['seed'] = [algo_seed] # Set seed for randomized optimization algorithm (I think...)
                num_seeds += 1
                
                clfRunner = mlh.NNGSRunner(
                                        x_train=x_train,
                                        y_train=y_train,
                                        x_test=x_test,
                                        y_test=y_test,
                                        experiment_name=experiment_name,
                                        algorithm=algorithm,
                                        grid_search_parameters=grid_search_parameters,
                                        grid_search_scorer_method=grid_search_scorer_method,
                                        iteration_list=np.round(np.logspace(0, 3.3, 20)),
                                        bias=True,
                                        early_stopping=True,
                                        clip_max=1e+10,
                                        max_attempts=10, # use default value
                                        cv=cv,
                                        generate_curves=True,
                                        seed=nn_seed,
                                        n_jobs=-1
                                        )
                
                run_stats_df, curves_df, cv_results_df, grid_search_cv = clfRunner.run()
                
                # Select what columns to keep track of
                df_my_stats = run_stats_df[keep_col]
                df_my_stats['nn_seed']     = nn_seed
                df_my_stats['algo_seed']   = algo_seed

                # Concatenate across all the seeded runs
                try:
                    df_my_stats_concat  = pd.concat([df_my_stats_concat, df_my_stats])
                except UnboundLocalError:
                    df_my_stats_concat = df_my_stats
        
        # Calculate summary statistics across the experiments
        groupby_list    = ['algorithm', 'Iteration']
        df_stats_mean   = df_my_stats_concat[keep_col].groupby(groupby_list).mean()
        df_stats_std    = df_my_stats_concat[keep_col].groupby(groupby_list).std()
        df_stats        = df_stats_mean.join(df_stats_std, lsuffix='_mean', rsuffix='_std')
        std_of_mean     = df_stats['Fitness_std']/(num_seeds-1)**0.5
        z_score         = 1.96 # 1.96 for 95% confidence
        df_stats['Fitness_std_of_mean'] = z_score*std_of_mean
        df_stats.reset_index(level='Iteration', inplace=True)
        
    return df_stats
    