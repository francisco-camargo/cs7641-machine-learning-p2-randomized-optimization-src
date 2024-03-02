"""
follow:
    https://github.com/hiive/mlrose/blob/master/nn_examples.ipynb

In order to use SyntheticData and plot_synthetic_dataset, I copied that code and make a file to contain these
functions within this repo because the version of mlrose-hiive that I am using does not have them
So I created the make_data.py file

In order to import SKMLPRunner with the version of mlrose-hiive that I have I had to go to the 
__init__.py file of mlrose-hiive and modify this line:
from .runners import GARunner, MIMICRunner, RHCRunner, SARunner, NNGSRunner

to now include SKMLPRunner:
from .runners import GARunner, MIMICRunner, RHCRunner, SARunner, NNGSRunner, SKMLPRunner
    
"""

# from IPython.core.display import display, HTML # for some notebook formatting.

import logging
from src.nn_synthetic_data.make_data import SyntheticData, plot_synthetic_dataset
# from make_data import SyntheticData, plot_synthetic_dataset

# switch off the chatter
logging.basicConfig(level=logging.WARNING)

# set seed for data sampling
sd = SyntheticData(seed=123456)


def get_sample_test_train(x_dim=20, y_dim=20, add_noise=0.0):
    noisy_data, noisy_features, noisy_classes, _ = sd.get_synthetic_data(x_dim, y_dim, add_noise)
    nx, ny, x_train, x_test, y_train, y_test = sd.setup_synthetic_data_test_train(noisy_data)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    
    # Generate data
    x_train, y_train, x_test, y_test = get_sample_test_train(5,5,0.0)
    # plot_synthetic_dataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, transparent_bg=False, bg_color='black')

    ########################################
    # Example where we fit the data
    import numpy as np
    import mlrose_hiive
    
    # ensure defaults are in grid search
    # default_grid_search_parameters = {
    #     'max_iters': [5000],
    #     'learning_rate_init': [0.1, 0.2, 0.4, 0.8],
    #     'hidden_layer_sizes': [[4, 4, 4]],
    #     'activation': [mlrose_hiive.neural.activation.relu],
    # }

    # default_parameters = {
    #     'seed': 123456,
    #     'iteration_list': 2 ** np.arange(13),
    #     'max_attempts': 5000,
    #     # 'override_ctrl_c_handler': False, # required for running in notebook
    #     'n_jobs':5,
    #     'cv':5,
    # }
    
    
    # skmlp_grid_search_parameters = {
    #     **default_grid_search_parameters,
    #     'max_iters': [5000],
    #     'learning_rate_init': [0.001],
    #     'activation': [mlrose_hiive.neural.activation.sigmoid],
    # }

    # skmlp_default_parameters = {
    #     **default_parameters,
    #     'early_stopping':True,
    #     'tol':1e-05,
    #     'alpha':0.001,
    #     'solver':'lbfgs',
    # }

    skmlp_grid_search_parameters = {
        'max_iters': [1000],
        'learning_rate_init': [0.01,0.1,0.2],
        'hidden_layer_sizes': [[4, 4]],
        'activation': [mlrose_hiive.neural.activation.relu],
    }

    skmlp_default_parameters = {
        'seed': 42,
        'iteration_list': np.round(np.logspace(0, 2.5, 20)),
        'max_attempts': 100,
        'n_jobs':5,
        'cv':2,
        'early_stopping':True,
        'tol':1e-03,
        'alpha':0.001,
        'solver':'lbfgs',
    }
    cx_skr = mlrose_hiive.SKMLPRunner(x_train=x_train, y_train=y_train,
                  x_test=x_test, y_test=y_test,
                  experiment_name='skmlp_clean',
                  grid_search_parameters = skmlp_grid_search_parameters,
                  **skmlp_default_parameters)

    run_stats_df, curves_df, cv_results_df, cx_sr = cx_skr.run() 
  
    # Plot the results
    print(f'Hidden layer size: {cx_sr.best_params_["hidden_layer_sizes"]}')
    plot_synthetic_dataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, classifier=cx_sr.best_estimator_.mlp)