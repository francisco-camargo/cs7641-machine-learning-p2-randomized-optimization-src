"""
following
    https://github.com/hiive/mlrose/blob/master/problem_examples.ipynb
    https://github.com/hiive/mlrose/blob/master/tutorial_examples.ipynb

alternative
    https://www.kaggle.com/code/makhil2008/cs7641-randomized-optimization-part-1

    N-Queens hyperparameters
        board_size

    Random Optimizer hyperparameters:
        seed
        iteration_list
        max_attempts

    Simulated Annealing hyperparameters:
        temperature_list
            schedule_type
                * ArithDecay
                * ExpDecay
                * GeomDecay
            init_temp
            decay
            min_temp

when starting on a new higher value of problem size
    set iteration_list and max_attempts to high values
    find the domain over which the algo specific HPs will get good performance
    in subsequent runs, reduce how generous we are with iteration_list and max_attempts

for fixed problem size and algo
    run experiment a bunch of times
    find set of algo HPs that resulted in good performance, this is the domain of good performance
    I'm guess that this domain will morph as problem size increases but overall it will shrink

"""

import os
import pandas as pd
import matplotlib.pyplot as plt

import src.hp_search.hp_search as hp_search
import src.vtime.vtime as vtime
import src.vsize.vsize as vsize
import src.plotting as plotting

# TODO: remake cpeaks plots with better y axis notation
# TODO: flip fitness axis of NN plots

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



def main(experiment_name, experiment_dict, mode, experiment_steps):
    problem_dict = experiment_dict['problem_dict']
    algo_dict = experiment_dict['algo_dict']

    folder_path = os.path.join('.', 'data', problem_dict['problem'])
    filename = f'{experiment_name}.csv'

    if mode == 'optimize':
        df_stats, _ = hp_search.run_hp_search(experiment_dict=experiment_dict)
        folder_path = os.path.join('.', 'data', problem_dict['problem'], algo_dict['algo'])
        filename = f'{experiment_name}__df_summary.csv'
        save_data(df_stats, folder_path, filename)

        # Start Plotting

        # Fitness vs Iterations, fixed problem size
        # plt.plot(df_curves['Iteration_mean'], df_curves['Fitness_mean'], 'o')
        unique_experiments = df_stats.index.unique()
        # styles = ['solid', 'dotted', 'dashed', 'dashdot']
        fig = plt.figure()
        for idx, u_exp in enumerate(unique_experiments):
            if idx <= 9:    linestyle = 'solid'
            elif idx <= 19: linestyle = 'dotted'
            elif idx <= 29: linestyle = 'dashed'
            elif idx <= 39: linestyle = 'dashdot'
            x = df_stats.xs(u_exp)['Iteration']
            y = df_stats.xs(u_exp)['Fitness_mean']
            plt.plot(x, y, label=str(u_exp), linestyle=linestyle)
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        # plt.xlim([1, 1e3])
        # plt.ylim([0, 5])
        # plt.legend(bbox_to_anchor=(1.1,1.05), loc='upper right')
        plt.legend()
        plt.show()

        # Fitness vs Time, fixed problem size
        fig = plt.figure()
        for idx, u_exp in enumerate(unique_experiments):
            if idx <= 9:    linestyle = 'solid'
            elif idx <= 19: linestyle = 'dotted'
            elif idx <= 29: linestyle = 'dashed'
            elif idx <= 39: linestyle = 'dashdot'
            x = df_stats.xs(u_exp)['Time_mean']
            y = df_stats.xs(u_exp)['Fitness_mean']
            plt.plot(x, y, label=str(u_exp), linestyle=linestyle)
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Time (s)')
        plt.ylabel('Fitness')
        # plt.xlim([1, 1e3])
        # plt.ylim([0, 5])
        # plt.legend(bbox_to_anchor=(1.1,1.05), loc='upper right')
        plt.legend()
        plt.show()

        # Fitness vs FEvals, fixed problem size
        fig = plt.figure()
        for idx, u_exp in enumerate(unique_experiments):
            if idx <= 9:    linestyle = 'solid'
            elif idx <= 19: linestyle = 'dotted'
            elif idx <= 29: linestyle = 'dashed'
            elif idx <= 39: linestyle = 'dashdot'
            x = df_stats.xs(u_exp)['FEvals_mean']
            y = df_stats.xs(u_exp)['Fitness_mean']
            plt.plot(x, y, label=str(u_exp), linestyle=linestyle)
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('Number of Function Evaluations')
        plt.ylabel('Fitness')
        # plt.xlim([1, 1e3])
        # plt.ylim([0, 5])
        # plt.legend(bbox_to_anchor=(1.1,1.05), loc='upper right')
        plt.legend()
        plt.show()

    elif mode == 'vtime':

        if 'run' in experiment_steps:
            df_stats, _ = vtime.run_vtime(experiment_dict=experiment_dict)
            save_data(df_stats, folder_path, filename)

        if 'plot' in experiment_steps:
            df = read_data(folder_path, filename)
            df.set_index(['problem', 'algo'], inplace=True)

            dependent_variable = 'Fitness_mean'
            dependent_variable_halfband = 'Fitness_std'
            show = True
            flip_values = True

            plotting.plot_fixed_problem_size(df=df.copy(),
                                             independent_variable='Iteration',
                                             dependent_variable=dependent_variable,
                                             dependent_variable_halfband=dependent_variable_halfband,
                                             show=show,
                                             flip_values=flip_values)

            plotting.plot_fixed_problem_size(df=df.copy(),
                                             independent_variable='Time_mean',
                                             dependent_variable=dependent_variable,
                                             dependent_variable_halfband=dependent_variable_halfband,
                                             show=show,
                                             flip_values=flip_values)

            plotting.plot_fixed_problem_size(df=df.copy(),
                                             independent_variable='FEvals_mean',
                                             dependent_variable=dependent_variable,
                                             dependent_variable_halfband=dependent_variable_halfband,
                                             show=show,
                                             flip_values=flip_values)

    elif mode == 'vsize':

        if 'run' in experiment_steps:
            df_stats, df_curves = vsize.run_vsize(experiment_dict=experiment_dict)
            save_data(df_stats, folder_path, filename)

        if 'plot' in experiment_steps:
            df = read_data(folder_path, filename)
            df.set_index(['problem', 'algo'], inplace=True)

            show = True
            flip_values=True

            dependent_variable = 'Fitness_mean'
            dependent_variable_halfband = 'Fitness_std_of_mean'
            plotting.plot_variable_problem_size(df=df.copy(),
                                             independent_variable='size',
                                             dependent_variable=dependent_variable,
                                             dependent_variable_halfband=dependent_variable_halfband,
                                             show=show,
                                             flip_values=flip_values)

            dependent_variable = 'Time_mean'
            dependent_variable_halfband = 'Time_std_of_mean'
            plotting.plot_variable_problem_size(df=df.copy(),
                                             independent_variable='size',
                                             dependent_variable=dependent_variable,
                                             dependent_variable_halfband=dependent_variable_halfband,
                                             show=show,
                                             legend_loc='upper left')

            dependent_variable = 'FEvals_mean'
            dependent_variable_halfband = 'FEvals_std_of_mean'
            plotting.plot_variable_problem_size(df=df.copy(),
                                             independent_variable='size',
                                             dependent_variable=dependent_variable,
                                             dependent_variable_halfband=dependent_variable_halfband,
                                             show=show,
                                             legend_loc='upper left')


if __name__ == '__main__':
    experiment_steps = ['plot'] # run, plot
    # mode = 'optimize'; experiment_name = 'nqueens_20_ga'
    # mode = 'vtime'; experiment_name = 'tsp_vtime'
    mode = 'vsize'; experiment_name = 'maxkcolor_vsize'

    # Queens
    if experiment_name == 'nqueens_4_rhc':
        import src.hp_search.nqueens_4_rhc as nqueens_4_rhc
        experiment_dict = nqueens_4_rhc.get_exp()
    if experiment_name == 'nqueens_6_rhc':
        import src.hp_search.nqueens_6_rhc as nqueens_6_rhc
        experiment_dict = nqueens_6_rhc.get_exp()
    if experiment_name == 'nqueens_20_rhc':
        import src.hp_search.nqueens_20_rhc as nqueens_20_rhc
        experiment_dict = nqueens_20_rhc.get_exp()

    if experiment_name == 'nqueens_4_sa':
        import src.hp_search.nqueens_4_sa as nqueens_4_sa
        experiment_dict = nqueens_4_sa.get_exp()
    if experiment_name == 'nqueens_5_sa':
        import src.hp_search.nqueens_5_sa as nqueens_5_sa
        experiment_dict = nqueens_5_sa.get_exp()
    if experiment_name == 'nqueens_6_sa':
        import src.hp_search.nqueens_6_sa as nqueens_6_sa
        experiment_dict = nqueens_6_sa.get_exp()
    if experiment_name == 'nqueens_20_sa':
        import src.hp_search.nqueens_20_sa as nqueens_20_sa
        experiment_dict = nqueens_20_sa.get_exp()

    if experiment_name == 'nqueens_6_ga':
        import src.hp_search.nqueens_6_ga as nqueens_6_ga
        experiment_dict = nqueens_6_ga.get_exp()
    if experiment_name == 'nqueens_20_ga':
        import src.hp_search.nqueens_20_ga as nqueens_20_ga
        experiment_dict = nqueens_20_ga.get_exp()

    if experiment_name == 'nqueens_20_mimic':
        import src.hp_search.nqueens_20_mimic as nqueens_20_mimic
        experiment_dict = nqueens_20_mimic.get_exp()

    if experiment_name == 'nqueens_vsize':
        import src.vsize.nqueens_vsize as nqueens
        experiment_dict = nqueens.get_exp()

    if experiment_name == 'nqueens_vtime':
        import src.vsize.nqueens_vsize as nqueens
        experiment_dict = nqueens.get_exp()



    # FlipFlop
    if experiment_name == 'flipflop_4_rhc':
        import src.hp_search.flipflop_4_rhc as flipflop_4_rhc
        experiment_dict = flipflop_4_rhc.get_exp()
    if experiment_name == 'flipflop_10_rhc':
        import src.hp_search.flipflop_10_rhc as flipflop_10_rhc
        experiment_dict = flipflop_10_rhc.get_exp()
    if experiment_name == 'flipflop_20_rhc':
        import src.hp_search.flipflop_20_rhc as flipflop_20_rhc
        experiment_dict = flipflop_20_rhc.get_exp()
    if experiment_name == 'flipflop_30_rhc':
        import src.hp_search.flipflop_30_rhc as flipflop_30_rhc
        experiment_dict = flipflop_30_rhc.get_exp()

    if experiment_name == 'flipflop_3_sa':
        import src.hp_search.flipflop_3_sa as flipflop_3_sa
        experiment_dict = flipflop_3_sa.get_exp()
    if experiment_name == 'flipflop_4_sa':
        import src.hp_search.flipflop_4_sa as flipflop_4_sa
        experiment_dict = flipflop_4_sa.get_exp()
    if experiment_name == 'flipflop_10_sa':
        import src.hp_search.flipflop_10_sa as flipflop_10_sa
        experiment_dict = flipflop_10_sa.get_exp()
    if experiment_name == 'flipflop_20_sa':
        import src.hp_search.flipflop_20_sa as flipflop_20_sa
        experiment_dict = flipflop_20_sa.get_exp()
    if experiment_name == 'flipflop_30_sa':
        import src.hp_search.flipflop_30_sa as flipflop_30_sa
        experiment_dict = flipflop_30_sa.get_exp()

    if experiment_name == 'flipflop_4_ga':
        import src.hp_search.flipflop_4_ga as flipflop_4_ga
        experiment_dict = flipflop_4_ga.get_exp()
    if experiment_name == 'flipflop_10_ga':
        import src.hp_search.flipflop_10_ga as flipflop_10_ga
        experiment_dict = flipflop_10_ga.get_exp()
    if experiment_name == 'flipflop_20_ga':
        import src.hp_search.flipflop_20_ga as flipflop_20_ga
        experiment_dict = flipflop_20_ga.get_exp()
    if experiment_name == 'flipflop_30_ga':
        import src.hp_search.flipflop_30_ga as flipflop_30_ga
        experiment_dict = flipflop_30_ga.get_exp()

    if experiment_name == 'flipflop_4_mimic':
        import src.hp_search.flipflop_4_mimic as flipflop_4_mimic
        experiment_dict = flipflop_4_mimic.get_exp()
    if experiment_name == 'flipflop_10_mimic':
        import src.hp_search.flipflop_10_mimic as flipflop_10_mimic
        experiment_dict = flipflop_10_mimic.get_exp()
    if experiment_name == 'flipflop_20_mimic':
        import src.hp_search.flipflop_20_mimic as flipflop_20_mimic
        experiment_dict = flipflop_20_mimic.get_exp()
    if experiment_name == 'flipflop_30_mimic':
        import src.hp_search.flipflop_30_mimic as flipflop_30_mimic
        experiment_dict = flipflop_30_mimic.get_exp()

    if experiment_name == 'flipflop_vtime':
        import src.vsize.flipflop_vsize as flipflop
        experiment_dict = flipflop.get_exp()
    if experiment_name == 'flipflop_vsize':
        import src.vsize.flipflop_vsize as flipflop
        experiment_dict = flipflop.get_exp()

    # Continuous Peaks
    if experiment_name == 'cpeaks_20_0p1_rhc':
        import src.hp_search.cpeaks_20_0p1_rhc as cpeaks_20_0p1_rhc
        experiment_dict = cpeaks_20_0p1_rhc.get_exp()
    if experiment_name == 'cpeaks_40_0p1_rhc':
        import src.hp_search.cpeaks_40_0p1_rhc as cpeaks_40_0p1_rhc
        experiment_dict = cpeaks_40_0p1_rhc.get_exp()

    if experiment_name == 'cpeaks_20_0p1_sa':
        import src.hp_search.cpeaks_20_0p1_sa as cpeaks_20_0p1_sa
        experiment_dict = cpeaks_20_0p1_sa.get_exp()

    if experiment_name == 'cpeaks_20_0p1_ga':
        import src.hp_search.cpeaks_20_0p1_ga as cpeaks_20_0p1_ga
        experiment_dict = cpeaks_20_0p1_ga.get_exp()

    if experiment_name == 'cpeaks_20_0p1_mimic':
        import src.hp_search.cpeaks_20_0p1_mimic as cpeaks_20_0p1_mimic
        experiment_dict = cpeaks_20_0p1_mimic.get_exp()
    if experiment_name == 'cpeaks_40_0p1_mimic':
        import src.hp_search.cpeaks_40_0p1_mimic as cpeaks_40_0p1_mimic
        experiment_dict = cpeaks_40_0p1_mimic.get_exp()

    if experiment_name == 'cpeaks_vtime':
        import src.vsize.cpeaks_vsize as cpeaks
        experiment_dict = cpeaks.get_exp()
    if experiment_name == 'cpeaks_vsize':
        import src.vsize.cpeaks_vsize as cpeaks
        experiment_dict = cpeaks.get_exp()

    # MaxKColor
    if experiment_name == 'maxkcolor_rhc':
        import src.hp_search.maxkcolor_rhc as maxkcolor_rhc
        experiment_dict = maxkcolor_rhc.get_exp()
    if experiment_name == 'maxkcolor_sa':
        import src.hp_search.maxkcolor_sa as maxkcolor_sa
        experiment_dict = maxkcolor_sa.get_exp()
    if experiment_name == 'maxkcolor_ga':
        import src.hp_search.maxkcolor_ga as maxkcolor_ga
        experiment_dict = maxkcolor_ga.get_exp()
    if experiment_name == 'maxkcolor_mimic':
        import src.hp_search.maxkcolor_mimic as maxkcolor_mimic
        experiment_dict = maxkcolor_mimic.get_exp()

    if experiment_name == 'maxkcolor_vtime' or experiment_name == 'maxkcolor_vsize':
        import src.vsize.maxkcolor_vsize as maxkcolor
        experiment_dict = maxkcolor.get_exp()

    # Traveling Salesmen Problem
    if experiment_name == 'tsp_10_rhc':
        import src.hp_search.tsp_10_rhc as tsp_10_rhc
        experiment_dict = tsp_10_rhc.get_exp()

    if experiment_name == 'tsp_10_sa':
        import src.hp_search.tsp_10_sa as tsp_10_sa
        experiment_dict = tsp_10_sa.get_exp()
    if experiment_name == 'tsp_20_sa':
        import src.hp_search.tsp_20_sa as tsp_20_sa
        experiment_dict = tsp_20_sa.get_exp()

    if experiment_name == 'tsp_10_ga':
        import src.hp_search.tsp_10_ga as tsp_10_ga
        experiment_dict = tsp_10_ga.get_exp()
    if experiment_name == 'tsp_20_ga':
        import src.hp_search.tsp_20_ga as tsp_20_ga
        experiment_dict = tsp_20_ga.get_exp()

    if experiment_name == 'tsp_10_mimic':
        import src.hp_search.tsp_10_mimic as tsp_10_mimic
        experiment_dict = tsp_10_mimic.get_exp()
    if experiment_name == 'tsp_20_mimic':
        import src.hp_search.tsp_20_mimic as tsp_20_mimic
        experiment_dict = tsp_20_mimic.get_exp()

    if experiment_name == 'tsp_vtime':
        import src.vsize.tsp_vsize as tsp
        experiment_dict = tsp.get_exp()
    if experiment_name == 'tsp_vsize':
        import src.vsize.tsp_vsize as tsp
        experiment_dict = tsp.get_exp()

    print(f'Run: {experiment_name}')
    main(experiment_name=experiment_name, experiment_dict=experiment_dict, mode=mode, experiment_steps=experiment_steps)
