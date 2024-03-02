"""

Plotting code for assignment 2

Given a problem and an algorithm,
Plot fitness    vs iteration
                vs clock time
                vs FEvals
                vs problem size

This means here we need to be able to read results per problem per algorithm and take it from there
"""

import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import colors

# determine transparent color equivalents
# https://stackoverflow.com/questions/33371939/calculate-rgb-equivalent-of-base-colors-with-alpha-of-0-5-over-white-background
def make_rgb_transparent(rgb, bg_rgb, alpha):
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]

fontsize = 9
fontsize_ticks = fontsize - 2
fig_dim_x = 3.2
fig_dim_y = fig_dim_x * 0.75
alpha=0.2

def plot_fixed_problem_size(df, independent_variable, dependent_variable, dependent_variable_halfband, legend_loc='best', show=False, flip_values=False):
    if independent_variable == 'Time_mean':
        x_label = independent_variable + ' (s)'
    else:
        x_label = independent_variable
    # plot fitness vs iterations at fixed problem size
    unique_experiments = df.index.unique()
    fig = plt.figure()
    fig.set_size_inches(fig_dim_x, fig_dim_y)
    for idx, u_exp in enumerate(unique_experiments):
        df_temp = df.xs(u_exp)
        x       = df_temp[independent_variable]
        y       = df_temp[dependent_variable]
        band    = df_temp[dependent_variable_halfband]
        
        if flip_values:
            y *= -1
            
        # Plot
        p     = plt.plot(x, y, label=str(u_exp))
        color = p[0].get_color() # get str value of color
        color = colors.colorConverter.to_rgb(color) # convert to tuple value of color
        plt.fill_between(x,y+band,y-band, color=make_rgb_transparent(color, (1,1,1), alpha=alpha))
        
    plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(dependent_variable, fontsize=fontsize)
    # plt.xlim([1, 1e3])
    # plt.ylim([0.1, 10])
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.tick_params(direction='in', which='both')
    plt.legend(loc=legend_loc, fontsize=fontsize_ticks)
    plt.tight_layout(pad=0)
    # fig.patch.set_facecolor('xkcd:mint green') # use to debug image sizing
    
    path = os.path.join('data', 'results', f'{u_exp[0]}_{dependent_variable}_vs_{independent_variable}.eps')
    plt.savefig(path)
    if show:
        plt.show()
        

def plot_variable_problem_size(df, independent_variable, dependent_variable, dependent_variable_halfband, legend_loc='best', show=False, flip_values=False):
    unique_experiments = df.index.unique()
    fig = plt.figure()
    fig.set_size_inches(fig_dim_x, fig_dim_y)
    for idx, u_exp in enumerate(unique_experiments):
        max_iter= df.xs(u_exp)['Iteration'].max()
        mask    = df.xs(u_exp)['Iteration']==max_iter
        df_temp = df.xs(u_exp)[mask]
        x       = df_temp[independent_variable]
        y       = df_temp[dependent_variable]
        band    = df_temp[dependent_variable_halfband]
        
        if flip_values:
            y *= -1
        
        # Plot
        p     = plt.plot(x, y, label=str(u_exp))
        color = p[0].get_color() # get str value of color
        color = colors.colorConverter.to_rgb(color) # convert to tuple value of color
        plt.fill_between(x,y+band,y-band, color=make_rgb_transparent(color, (1,1,1), alpha=alpha))
        
        
    x_label = independent_variable
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Problem Size', fontsize=fontsize)
    plt.ylabel(dependent_variable, fontsize=fontsize)
    # plt.xlim([1, 1e3])
    # plt.ylim([0.1, 10])
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        # Don't show decimals in y axis
        # https://stackoverflow.com/questions/29188757/specify-format-of-floats-for-tick-labels
    plt.tick_params(direction='in', which='both')
    plt.legend(loc=legend_loc, fontsize=fontsize_ticks)
    plt.tight_layout(pad=0)
    # fig.patch.set_facecolor('xkcd:mint green') # use to debug image sizing
    
    path = os.path.join('data', 'results', f'{u_exp[0]}_{dependent_variable}_vs_{independent_variable}.eps')
    plt.savefig(path)
    if show:
        plt.show()
        