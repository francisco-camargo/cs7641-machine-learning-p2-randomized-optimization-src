import os
import pandas as pd

import src.data_prep as data_prep
import src.experiments.optimize_experiment as optimize_experiment
import src.experiments.fitness_experiment as fitness_experiment
import src.experiments.learning_curve_experiment as learning_curve_experiment
import src.plotting as plotting


# TODO: explain how to run code in README.txt


def save_data(df, folder_path, filename):
    print('Save data')
    try: os.makedirs(folder_path)
    except FileExistsError: pass # if this folder already exists, no need to remake it
    path = os.path.join(folder_path, filename)
    df.to_csv(path)

def read_data(folder_path, filename):
    print('Read data')
    path = os.path.join(folder_path, filename)
    df = pd.read_csv(path)
    return df

def main(
        problem_name,
        experiment_mode,
        optimize_algo,
        fitness_algos,
        learning_algos,
    ):
    
    # Data prep
    x_train, y_train, x_test, y_test = data_prep.run_prep(problem_name=problem_name)
    import sklearn.metrics as metrics
    grid_search_scorer_method=metrics.f1_score

    # TODO: These experiments only support the MONKs problem at the moment
    if experiment_mode=='optimize':
        optimize_experiment.run_experiment(optimize_algo, grid_search_scorer_method, x_train, y_train, x_test, y_test)
    
    if experiment_mode=='fitness':
        folder_path = os.path.join('.', 'data', 'results')
        filename = f'{experiment_mode}__df_summary.csv'
        
        # experiment code
        df_stats = fitness_experiment.run_experiment(fitness_algos, grid_search_scorer_method, x_train, y_train, x_test, y_test)
        
        # save data
        save_data(df_stats, folder_path, filename)
        
        df = read_data(folder_path, filename)
        
        # prep for plotting
        df.reset_index(inplace=True)
        df.set_index(['algorithm'], inplace=True)
        
        show = True
        plotting.plot_Fitness(df=df.copy(), independent_variable='Iteration', dependent_variable='Fitness_mean', dependent_variable_halfband='Fitness_std_of_mean', legend_loc='center left', show=show)
        plotting.plot_Fitness(df=df.copy(), independent_variable='Time_mean', dependent_variable='Fitness_mean', dependent_variable_halfband='Fitness_std_of_mean', legend_loc='center left', show=show)
        # plotting.plot_Fitness(df=df.copy(), independent_variable='FEvals_mean', dependent_variable='Fitness_mean', dependent_variable_halfband='Fitness_std_of_mean', legend_loc='upper left', show=show)
            # Values of FEvals being returned are None
        
    if experiment_mode=='learning':
        # experiment code
        df = learning_curve_experiment.run_experiment(learning_algos, grid_search_scorer_method, x_train, y_train, x_test, y_test)

        # save data
        folder_path = os.path.join('.', 'data', 'results')
        filename = f'{experiment_mode}__df_learning_curve.csv'
        save_data(df, folder_path, filename)
        
        plotting.plot_learning_curve(df=df)
        

if __name__ == '__main__':
    problem_name    = 'monks'
    experiment_mode = 'fitness' # optimize, fitness, learning
    
    optimize_algo   = 'gd'
    
    fitness_algos = [
        'gd',
        'rhc',
        'sa',
        'ga',
    ]
    
    learning_algos = [
        'gd',
        'rhc',
        'sa',
        'ga',
    ]
        
    print(f'Run: {experiment_mode} experiment on {problem_name} problem')
    main(
        problem_name=problem_name,
        experiment_mode=experiment_mode,
        optimize_algo=optimize_algo,
        fitness_algos=fitness_algos,
        learning_algos=learning_algos,
        )
    