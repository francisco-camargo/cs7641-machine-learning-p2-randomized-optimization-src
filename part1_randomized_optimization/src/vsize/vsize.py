'''
    Given optimal parameters per algorithm for one specific problem. Solve that problem as a function of problem size
'''

import pandas as pd
from mlrose_hiive import QueensGenerator, FlipFlopGenerator, ContinuousPeaksGenerator, MaxKColorGenerator, TSPGenerator, KnapsackGenerator
from mlrose_hiive import RHCRunner, SARunner, GARunner, MIMICRunner, NNGSRunner
import src.helper_funcs as hf

def run_vsize(experiment_dict):
    problem_dict = experiment_dict['problem_dict']
    rand_optimization_dict = experiment_dict['rand_optimization_dict']
    algo_dict = experiment_dict['algo_dict']

    for algo in algo_dict:
        print(f'Current algo: {algo}')
        for size in problem_dict['problem_size']:
            int_size=int(size)
            print(f'\tCurrent size: {size}')
            for current_seed in rand_optimization_dict['seed']:
                print(f'\t\tCurrent seed: {current_seed}')
                groupby_list = ['problem', 'algo', 'Iteration', 'size']

                if problem_dict['problem'] == 'queens':
                    problem = QueensGenerator().generate(seed=current_seed, size=int_size)
                elif problem_dict['problem'] == 'flipflop':
                    problem = FlipFlopGenerator().generate(seed=current_seed, size=int_size)
                elif problem_dict['problem'] == 'cpeaks':
                    t_pct = problem_dict['t_pct']
                    problem = ContinuousPeaksGenerator().generate(seed=current_seed, size=int_size, t_pct=t_pct)
                elif problem_dict['problem'] == 'tsp':
                    area_width = 250
                    area_height = 250
                    problem = TSPGenerator().generate(seed=current_seed, number_of_cities=int_size, area_width=area_width, area_height=area_height)
                elif problem_dict['problem'] == 'maxkcolor':
                    problem = MaxKColorGenerator().generate(seed=current_seed, number_of_nodes=int_size)

                # instantiate runner
                if algo == 'rhc':
                    runner_obj = RHCRunner(
                            problem=problem,
                            experiment_name='',
                            seed=current_seed,
                            iteration_list=rand_optimization_dict['iteration_list'],
                            restart_list=algo_dict[algo]['restart_list'],
                            max_attempts=rand_optimization_dict['max_attempts'],
                            )
                elif algo == 'sa':
                    runner_obj = SARunner(
                            problem=problem,
                            # experiment_name=pathing_dict['experiment_name'],
                            experiment_name='',
                            # output_directory=pathing_dict['output_directory'],
                            seed=current_seed,
                            iteration_list=rand_optimization_dict['iteration_list'],
                            max_attempts=rand_optimization_dict['max_attempts'],
                            temperature_list=algo_dict[algo]['temperature_list'],
                            )
                elif algo == 'ga':
                    runner_obj = GARunner(
                            problem=problem,
                            experiment_name='',
                            seed=current_seed,
                            iteration_list=rand_optimization_dict['iteration_list'],
                            population_sizes=algo_dict[algo]['population_sizes'],
                            mutation_rates=algo_dict[algo]['mutation_rates'],
                            max_attempts=rand_optimization_dict['max_attempts'],
                            )
                elif algo == 'mimic':
                    runner_obj = MIMICRunner(
                            problem=problem,
                            experiment_name='',
                            seed=current_seed,
                            iteration_list=rand_optimization_dict['iteration_list'],
                            population_sizes=algo_dict[algo]['population_sizes'],
                            keep_percent_list=algo_dict[algo]['keep_percent_list'],
                            max_attempts=rand_optimization_dict['max_attempts'],
                            use_fast_mimic=False,
                            )

                # look for solutions
                df_run_stats_temp, df_run_curves_temp = runner_obj.run()
                    # I think df_run_states gives me data snapshots when the Iteration equals a certain number
                        # if we hit the max_attempts cap, the row corresponding to Iterations after this cap just feedforward the capped values... I think
                    # I think df_run_curves gives me less columns but lets me look at all the Iterations
            
                # if algo_dict['algo'] == 'sa':
                #     # Consolidate columns; some schedule_types use schedule_decay column, some use schedule_exp_const
                #     df_run_stats_temp['new_decay'] = df_run_stats_temp.apply(lambda row: hf.new_decay(row), axis=1)
                #     try:
                #         df_run_stats_temp.drop(columns=['schedule_decay'])
                #     except KeyError: pass
                #     try:
                #         df_run_stats_temp.drop(columns=['schedule_exp_const'])
                #     except KeyError: pass
                        
                # Keep track of the seed in the output data
                hf.add_data_values(df_run_curves_temp, experiment_dict, algo, current_seed)
                hf.add_data_values(df_run_stats_temp, experiment_dict, algo, current_seed)
                df_run_curves_temp['size'] = int_size
                df_run_stats_temp['size'] = int_size

                # Concatenate across all the seeded runs
                try:
                    df_run_curves = pd.concat([df_run_curves, df_run_curves_temp])
                    df_run_stats  = pd.concat([df_run_stats, df_run_stats_temp])
                except UnboundLocalError:
                    df_run_curves = df_run_curves_temp
                    df_run_stats = df_run_stats_temp
            
            df_run_curves_mean = df_run_curves.groupby([df_run_curves.index]).mean()
            df_run_curves_std  = df_run_curves.groupby([df_run_curves.index]).std()
            # df_stats_mean = df_run_stats.groupby([df_run_stats.index]).mean()
            # df_stats_std = df_run_stats.groupby([df_run_stats.index]).std()
            df_stats_mean = df_run_stats.groupby(groupby_list).mean()
            df_stats_std = df_run_stats.groupby(groupby_list).std()

            df_curves = df_run_curves_mean.join(df_run_curves_std, lsuffix='_mean', rsuffix='_std')
            df_stats  = df_stats_mean.join(df_stats_std, lsuffix='_mean', rsuffix='_std')

            # std_of_mean     = df_stats['Fitness_std']/(rand_optimization_dict['seed']-1)**0.5
            # df_stats['Fitness_std_of_mean'] = z_score*std_of_mean
            num_samples = len(rand_optimization_dict['seed'])
            df_stats['Fitness_std_of_mean'] = hf.add_confidence_halfband(df_stats['Fitness_std'], num_samples=num_samples)
            df_stats['Time_std_of_mean']    = hf.add_confidence_halfband(df_stats['Time_std'], num_samples=num_samples)
            df_stats['FEvals_std_of_mean']  = hf.add_confidence_halfband(df_stats['FEvals_std'], num_samples=num_samples)
            
            df_stats.reset_index(level='Iteration', inplace=True)

    return df_stats, df_curves
