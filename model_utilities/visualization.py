import matplotlib.pyplot as plt
import random
import pandas as pd 
import numpy as np

# Plot entire ensemble
def plotting_ensemble(growth_curve_fit, title, nyears, frac_plot=1, plotdata=False): 
    
    # get list of ensemble members and unique maximum sources
    key_list = list(growth_curve_fit.monte_carlo_dfs.keys())
    list_unique = growth_curve_fit.max_df.Source.unique()
    
    # define colors for plotting
    #color_list = ['grey','#bab86c','#6cbab8']  #6e7f80, 536878 #
    color_list = plt.cm.tab10.colors
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax.set_title(title, fontsize=15)
    ax.set_ylabel('AGB+BGB estimate (tCO2e/ha)', fontsize=15)
    ax.set_xlabel('Age', fontsize=15)
    
    for ic in range(0, len(growth_curve_fit.monte_carlo_dfs.keys())):
        df_here = growth_curve_fit.monte_carlo_dfs[key_list[ic]]
        # get number of ensemble members before filtering 
        n_members_orig = df_here.shape[1]
        # filter out unrealistic ensemble members, end lower than start
        df_here = df_here.loc[:, df_here.iloc[0] < df_here.iloc[99]]  #keep columns where row 0 < row 99
        n_members_filtered = n_members_orig - df_here.shape[1]
        print('Number ensemble members filtered: ', n_members_filtered, 'in ', str(ic))
        # number of columns to plot defined by fraction
        n_col = round(frac_plot * df_here.shape[1])
        # randomly select that number of columns
        cols_to_plot = random.sample(range(df_here.shape[1]), min(n_col, df_here.shape[1]))
        df_here = df_here.iloc[:,cols_to_plot]
        # get source name from col name
        source_here = key_list[ic].split('_')[0]
        # define color from source name
        color_id = np.where(list_unique == source_here)[0][0]
        plt.plot(df_here.index[0:nyears]+1, df_here.iloc[0:nyears,], alpha=0.1, color=color_list[color_id], 
                 label=source_here)
    
    # add data points if plotdata is true    
    if plotdata == True:
        groups = growth_curve_fit.growth_df.groupby("source")
        for name, group in groups:
            ax.plot(group["age"], group["agb_bgb_tCO2e_ha"], marker="o", linestyle="", label=name, alpha=0.5)
            
    # avoid repeated items in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
        
    return plt 

# define plotting functions
# plot just the maxs
def plotting_maxs(growth_curve_fit, title, nyears=100, plotdata=False):
    import matplotlib.lines as mlines
    prediction_df = growth_curve_fit.prediction_df
    prediction_df.index = prediction_df['Age']
    prediction_df = prediction_df.drop(['Age'], axis=1)
    
    #color_list = ['grey','#bab86c','#6cbab8']  #6e7f80, 536878 #
    color_list = plt.cm.tab10.colors
    
    fig, ax = plt.subplots(figsize=(15,5))
    ax.set_title(title, fontsize=15)
    ax.set_ylabel('AGB+BGB estimate (tCO2e/ha)', fontsize=15)
    ax.set_xlabel('Age', fontsize=15)
    
    #for ic in range(0, len(growth_curve_fit.max_df.Source.unique())):
    for ic in growth_curve_fit.max_df.Source.unique():
        cols_plot_here = [col for col in prediction_df.columns if str(ic) in col]
        color_id = np.where(growth_curve_fit.max_df.Source.unique() == ic)[0][0]
        ax.plot(prediction_df.index[1:nyears], prediction_df[cols_plot_here].iloc[1:nyears], 
                color=color_list[color_id], label=ic) #label=growth_curve_fit.max_df.Source.unique()[ic]
    
    if plotdata == True:
        groups = growth_curve_fit.growth_df.groupby("source")
        for name, group in groups:
            ax.plot(group["age"], group["agb_bgb_tCO2e_ha"], marker="o", linestyle="", label=name, alpha=0.5)
    
    # avoid repeated items in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    return plt