
import numpy as np

def add_data_values(df, experiment_dict, algo, current_seed):
    for key in experiment_dict['problem_dict']:
        temp = experiment_dict['problem_dict'][key]
        # print(key)
        # print(temp)
        if isinstance(temp, int) or isinstance(temp,float) or isinstance(temp, str):
            df[key] = temp
    df['seed'] = current_seed
    df['algo'] = algo

def new_decay(row):
    try:
        row['schedule_decay']
    except KeyError:
        return row['schedule_exp_const']
    try:
        row['schedule_exp_const']
    except KeyError:
        return row['schedule_decay']

    if not np.isnan(row['schedule_decay']):
        output = row['schedule_decay']
    else:
        output = row['schedule_exp_const']
    return output


from functools import wraps
import time
def timeit(func):
    '''
    https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    Use this function as a decorator
    
    example:
    
    @timeit
    def calculate_something(num):
        """
        Simple function that returns sum of all numbers up to the square of num.
        """
        total = sum((x for x in range(0, num**2)))
        return total
        
    '''
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'Function took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def add_confidence_halfband(x, num_samples, z_score=1.96): # 1.96 z-score for 95% confidence
    std_of_mean = x/(num_samples-1)**0.5
    return z_score*std_of_mean