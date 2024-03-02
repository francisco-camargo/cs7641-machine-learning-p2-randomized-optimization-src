import pandas as pd
from mlrose_hiive import QueensGenerator, FlipFlopGenerator, ContinuousPeaksGenerator, MaxKColorGenerator, TSPGenerator, KnapsackGenerator
from mlrose_hiive import RHCRunner, SARunner, GARunner, MIMICRunner, NNGSRunner
from ..helper_funcs import add_data_values, new_decay

def run_hp_search(experiment_dict):
    rand_optimization_dict = experiment_dict['rand_optimization_dict']
    problem_dict = experiment_dict['problem_dict']
    algo_dict = experiment_dict['algo_dict']

    # Define the problem
    for current_seed in rand_optimization_dict['seed']:
        print(f'Current seed: {current_seed}')
        groupby_list = ['problem', 'algo']

        if problem_dict['problem'] == 'queens':
            groupby_list += ['problem_size']
            problem_size = problem_dict['problem_size']
            problem = QueensGenerator().generate(seed=current_seed, size=int(problem_size))
        
        elif problem_dict['problem'] == 'flipflop':
            groupby_list += ['problem_size']
            problem_size = problem_dict['problem_size']
            problem = FlipFlopGenerator().generate(seed=current_seed, size=problem_size)
        
        elif problem_dict['problem'] == 'cpeaks':
            groupby_list += ['problem_size', 't_pct']
            problem_size = problem_dict['problem_size']
            t_pct = problem_dict['t_pct']
            problem = ContinuousPeaksGenerator().generate(seed=current_seed, size=problem_size, t_pct=t_pct)

        elif problem_dict['problem'] == 'maxkcolor':
            # groupby_list += ['number_of_nodes', 'max_connections_per_node', 'max_colors']
            groupby_list += ['problem_size']
            problem_size = int(problem_dict['problem_size'])
            # max_connections_per_node = problem_dict['max_connections_per_node']
            # max_colors = problem_dict['max_colors']
            # problem = MaxKColorGenerator().generate(seed=current_seed, number_of_nodes=number_of_nodes, max_connections_per_node=max_connections_per_node, max_colors=max_colors)
            problem = MaxKColorGenerator().generate(seed=current_seed, number_of_nodes=problem_size)

        elif problem_dict['problem'] == 'tsp':
            int_size = int(problem_dict['problem_size'])
            groupby_list += ['problem_size']
            area_width = 250
            area_height = 250
            problem = TSPGenerator().generate(seed=current_seed, number_of_cities=int_size, area_width=area_width, area_height=area_height)

        # Initial problem state
        state = problem.get_state()

        # instantiate runner
        if algo_dict['algo'] == 'rhc':
            groupby_list += ['Iteration', 'Restarts']
            runner_obj = RHCRunner(
                    problem=problem,
                    experiment_name='',
                    seed=current_seed,
                    iteration_list=rand_optimization_dict['iteration_list'],
                    restart_list=algo_dict['restart_list'],
                    max_attempts=rand_optimization_dict['max_attempts'],
                    )
        elif algo_dict['algo'] == 'sa':
            groupby_list += ['Iteration', 'schedule_type', 'schedule_init_temp', 'new_decay', 'schedule_min_temp']
            runner_obj = SARunner(
                    problem=problem,
                    # experiment_name=pathing_dict['experiment_name'],
                    experiment_name='',
                    # output_directory=pathing_dict['output_directory'],
                    seed=current_seed,
                    iteration_list=rand_optimization_dict['iteration_list'],
                    max_attempts=rand_optimization_dict['max_attempts'],
                    temperature_list=algo_dict['temperature_list'],
                    )
        elif algo_dict['algo'] == 'ga':
            groupby_list += ['Iteration', 'Population Size', 'Mutation Rate']
            runner_obj = GARunner(
                    problem=problem,
                    experiment_name='',
                    seed=current_seed,
                    iteration_list=rand_optimization_dict['iteration_list'],
                    population_sizes=algo_dict['population_sizes'],
                    mutation_rates=algo_dict['mutation_rates'],
                    max_attempts=rand_optimization_dict['max_attempts'],
                    )
        elif algo_dict['algo'] == 'mimic':
            groupby_list += ['Iteration','Population Size', 'Keep Percent']
            runner_obj = MIMICRunner(
                    problem=problem,
                    experiment_name='',
                    seed=current_seed,
                    iteration_list=rand_optimization_dict['iteration_list'],
                    population_sizes=algo_dict['population_sizes'],
                    keep_percent_list=algo_dict['keep_percent_list'],
                    max_attempts=rand_optimization_dict['max_attempts'],
                    use_fast_mimic=False,
                    )

        # look for solutions
        df_run_stats_temp, df_run_curves_temp = runner_obj.run()
            # I think df_run_states gives me data snapshots when the Iteration equals a certain number
                # if we hit the max_attempts cap, the row corresponding to Iterations after this cap just feedforward the capped values... I think
            # I think df_run_curves gives me less columns but lets me look at all the Iterations
    
        if algo_dict['algo'] == 'sa':
            # Consolidate columns; some schedule_types use schedule_decay column, some use schedule_exp_const
            df_run_stats_temp['new_decay'] = df_run_stats_temp.apply(lambda row: new_decay(row), axis=1)
            try:
                df_run_stats_temp.drop(columns=['schedule_decay'])
            except KeyError: pass
            try:
                df_run_stats_temp.drop(columns=['schedule_exp_const'])
            except KeyError: pass
                
        # Keep track of the seed in the output data
        add_data_values(df_run_curves_temp, experiment_dict, algo_dict['algo'], current_seed)
        add_data_values(df_run_stats_temp,  experiment_dict, algo_dict['algo'], current_seed)

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
    df_stats_mean = df_run_stats.groupby(groupby_list).mean() # TODO: problem if max_colors = None
    df_stats_std = df_run_stats.groupby(groupby_list).std()

    df_curves = df_run_curves_mean.join(df_run_curves_std, lsuffix='_mean', rsuffix='_std')
    df_stats  = df_stats_mean.join(df_stats_std, lsuffix='_mean', rsuffix='_std')
    df_stats.reset_index(level='Iteration', inplace=True)

    return df_stats, df_curves
