import os
import matplotlib.pyplot as plt
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
    
def plot_Fitness(df, independent_variable, dependent_variable, dependent_variable_halfband, legend_loc='best', show=False):
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
        y       = -df_temp[dependent_variable]
        band    = df_temp[dependent_variable_halfband]
        
        # Plot
        p     = plt.plot(x, y, label=str(u_exp))
        color = p[0].get_color() # get str value of color
        color = colors.colorConverter.to_rgb(color) # convert to tuple value of color
        plt.fill_between(x,y+band,y-band, color=make_rgb_transparent(color, (1,1,1), alpha=alpha))
        
    plt.xscale('log')
    plt.yscale('symlog')
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(dependent_variable, fontsize=fontsize)
    # plt.xlim([1, 1e3])
    plt.ylim([-2e1, 0])
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.tick_params(direction='in', which='both')
    plt.legend(loc=legend_loc, fontsize=fontsize_ticks)
    plt.tight_layout(pad=0)
    # fig.patch.set_facecolor('xkcd:mint green') # use to debug image sizing
    
    path = os.path.join('data', 'results', f'{independent_variable}.eps')
    plt.savefig(path)
    if show:
        plt.show()
        
def plot_learning_curve(df, legend_loc='best', show=False):

    # plot fitness vs iterations at fixed problem size
    unique_experiments = df.index.unique()

    for idx, u_exp in enumerate(unique_experiments):
        
        fig = plt.figure()
        fig.set_size_inches(fig_dim_x, fig_dim_y)
    
        df_temp     = df.xs(u_exp)
        
        x           = df_temp['train_size']
        y_train     = df_temp['mean_train_score']
        train_band  = df_temp['std_of_mean_train_score']
        y_test      = df_temp['mean_test_score']
        test_band   = df_temp['std_of_mean_test_score']
        
        # Plot test
        color = 'green'
        p     = plt.plot(x, y_test, label=f'{u_exp} test', color=color)
        # color = p[0].get_color() # get str value of color
        color = colors.colorConverter.to_rgb(color) # convert to tuple value of color
        plt.fill_between(x,y_test +test_band, y_test -test_band, color=make_rgb_transparent(color, (1,1,1), alpha=alpha))
        
        # Plot training
        color = 'red'
        p     = plt.plot(x, y_train, label=f'{u_exp} training', color=color)
        # color = p[0].get_color() # get str value of color
        color = colors.colorConverter.to_rgb(color) # convert to tuple value of color
        plt.fill_between(x,y_train+train_band,y_train-train_band, color=make_rgb_transparent(color, (1,1,1), alpha=alpha))
        
        plt.xlabel('Training Data Size', fontsize=fontsize)
        plt.ylabel('f1-score', fontsize=fontsize)
        # plt.xlim([1, 1e3])
        plt.ylim([0.2, 1.0])
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.tick_params(direction='in', which='both')
        plt.legend(loc=legend_loc, fontsize=fontsize_ticks)
        plt.tight_layout(pad=0)
        
        # Save figures
        path = os.path.join('data', 'results', f'{u_exp}_learning_curve.eps')
        plt.savefig(path)
        
        if show:
            plt.show()
    