'''
Code to run explore hyperparameter space
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
    
    # By definition of what function this is, we want to use the appropriate hyperparameters to make a learning curve
    mode='learning'
    cv = 20
    num_size_points = 10
    
    keep_cols = ['mean_fit_time', 'mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score']
    
    ########################################
    # Now run the experiments
    for algo_nickname in algorithm_list: # Pick optimization algorithm
        print()
        print(f'Use algorithm: {algo_nickname}')
        
        # Get hyperparameters
        exp_out = helpers.exp_setup(algo_nickname=algo_nickname, mode=mode)
        algorithm, _, grid_search_parameters, nn_seed_iterable, algo_seed_iterable = exp_out
    
        for nn_seed in [1]:
            print()
            print(f'\tnn_seed: {nn_seed}')
            for algo_seed in [1]:
                print(f'\t\talgo_seed: {algo_seed}')
                grid_search_parameters['seed'] = [algo_seed]
                
                # resize data
                training_size = np.linspace(0.1, 1.0, num_size_points)
                bare_size = len(y_train)
                training_size = map(int, np.floor(training_size*bare_size) )
                
                for new_size in training_size:
                    x_train_new = x_train[:new_size]
                    y_train_new = y_train[:new_size]
                    # print(len(y_train))
                    clfRunner = mlh.NNGSRunner(
                                            x_train=x_train_new,
                                            y_train=y_train_new,
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

                    df = cv_results_df[keep_cols]
                    df['algorithm'] = algo_nickname
                    df['nn_seed']   = nn_seed
                    df['algo_seed'] = algo_seed
                    df['train_size']= new_size
                    
                    z_score         = 1.96 # 1.96 for 95% confidence
                    train     = df['std_train_score']/(cv-1)**0.5
                    test      = df['std_test_score']/(cv-1)**0.5
                    
                    df['std_of_mean_train_score'] = z_score*train
                    df['std_of_mean_test_score']  = z_score*test
                    
                    # Concatenate across all the seeded runs
                    try:
                        df_concat  = pd.concat([df_concat, df])
                    except UnboundLocalError:
                        df_concat = df
    
    df_concat.reset_index(inplace=True, drop=True)
    df_concat.set_index(['algorithm'], inplace=True)
    
    return df_concat