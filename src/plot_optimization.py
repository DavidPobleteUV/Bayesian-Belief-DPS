#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:39:45 2024
Last update on Fri April 16 16:17:20 2024
@author: mofanz
"""


import os
import matplotlib
import numpy as np
import pandas as pd
from pipe_simulation import Pipe
from pipesim_individual import Pipe_sim
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy.matlib as mat
from platypus.indicators import Hypervolume
import json
from utils_indicators import *
from policy import *
import matplotlib.patches as mpatches


def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    

def plot_pareto(obj, nseeds, config_name):

    #plt.style.use('seaborn-darkgrid')
    sns.set_style('darkgrid')
    colors = ['blue', 'green', 'red', 'purple', 'orange'] 

    if nseeds == 1:
        mask = is_pareto_efficient(np.array(obj))
        objs = []
        for m, o in zip(mask, obj):
            if m:
                objs.append(o)
        
        plt.scatter([o[0] for o in objs],[o[1] for o in objs])
        plt.xlabel('JPlanning')
        plt.ylabel('Jdeficit')
        #plt.xlim([0, 1500])
        #plt.ylim([0, 20])
    else:        
        for s in range(nseeds): #assuming multiple runs    
            #plt.scatter([o[0] for o in objs[s]],[o[1] for o in objs[s]])
            seed_obj = obj[s] 
            print(f"Seed {s+1} objectives shape: {np.array(seed_obj).shape}")  # Debugging output
            if np.array(seed_obj).ndim != 2 or np.array(seed_obj).shape[1] == 0:
                print("Skipping seed due to incorrect shape or empty objectives.")
                continue
            # Calculate Pareto efficiency
            mask = is_pareto_efficient(np.array(seed_obj))
            pareto_points = [o for m, o in zip(mask, seed_obj) if m]
            if pareto_points:  # Only attempt to plot if there are Pareto points
                plt.scatter(*zip(*pareto_points), color=colors[s % len(colors)], label=f'Seed {s+1}')

        #plt.xlim([0, 1500])
        #plt.ylim([0, 1500])
        plt.xlabel('JPlanning')
        plt.ylabel('Jdeficit')
    plt.title('Poreto objs') 
    plt.savefig(f'results/figures/{config_name}_pareto_objs.png')   
    plt.show()
            
def plot_pareto_compare(bayesDPS_objs, normDPS_objs, config_bayesDPS, config_normDPS,label):
    
    #Plots and saves a comparison of Pareto frontiers for BayesDPS and NormDPS.
    folder_path = f'results/figures/{label}/Pareto_frontiers'
    os.makedirs(folder_path, exist_ok=True)

   
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    plt.scatter(*zip(*bayesDPS_objs), c = 'blue', s = 100, label='BayesDPS ')
    plt.scatter(*zip(*normDPS_objs), c= 'pink', s = 100, label='NormDPS')
    plt.title(f'{label}_Pareto Frontiers under training climates')
    plt.xlabel('JPlanning')
    plt.ylabel('Jdeficit')
    plt.legend()
    plt.grid(True)

    # Save the plot
    file_name = f'training_pareto comparision.png'
    full_path = os.path.join(folder_path, file_name)
    plt.savefig(full_path, dpi=100)
    plt.show()

# Function to plot Pareto frontiers from evaluating optimal policies under test set climates
def plot_test_pareto(bayes_objs, norm_objs ,label, sim_label):
    """
    Plot the Pareto frontiers from evaluating optimal policies under test set cliamtes
    
    Parameters:
    - bayes_objs: List of objectives from evaluating optimal policies under test set climates using BayesDPS
    - norm_objs: List of objectives from evaluating optimal policies under test set climates using NormDPS
    - label: Label for the plot
    - sim_label: Label for the simulation (e.g., 'test_all', 'test_wet')
 
    """
    folder_path = f'results/figures/{label}/Pareto_frontiers'
    os.makedirs(folder_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the objectives from the BayesDPS
    bayes_objs = np.array(bayes_objs)
    ax.scatter(bayes_objs[:, 0], bayes_objs[:, 1], c='blue', label='BayesDPS', s = 100)

    # Plot the objectives from the NormDPS
    norm_objs = np.array(norm_objs)
    ax.scatter(norm_objs[:, 0], norm_objs[:, 1], c='pink', label='NormDPS', s = 100)

    # Set the overall font size for plots
    plt.rcParams.update({'font.size': 14})
    ax.set_xlabel('Planning Cost (Jplan) [M$]', fontsize=14)
    ax.set_ylabel('Deficit Cost (Jdef)', fontsize=14)
    ax.set_title(f'{label}_Pareto Frontiers under_{sim_label}_ climates', fontsize=16)
    ax.legend()
    ax.grid(True)
    # Save the plot
    file_name = f'{sim_label}_pareto_comparison.png'
    full_path = os.path.join(folder_path, file_name)
    plt.savefig(full_path, dpi=100)
    plt.show()



# plot precipitation, inflow and annual deficit
def plot_climate_series(nsim, sim_label, climate_label, climate_classes, label, save_plot=True):
    """
    Plot the annual precipitation, inflow, and deficit for each climate classification.
    Parameters:
    - nsim: List of model indices to plot
    - sim_label: Label for the simulation (e.g., 'training', 'test_wet')
    - climate_label: Label for the climate data (e.g., 'test', 'test3')
    - climate_classes: List of climate classifications for each model
    - label: Label for the plot
    """

    folder_path = f'results/figures/{label}/climate_time_series'
    os.makedirs(folder_path, exist_ok=True)

    data_loc = 'data/Annual_GPR'
    H = 100 * 12  # Number of months

    prep_mat = f'{climate_label}_month_prep.csv'
    preps = pd.read_csv(f'{data_loc}/{prep_mat}', header=None, skiprows=1)

    inflow_mat = f'{climate_label}_runoff.csv'
    inflows = pd.read_csv(f'{data_loc}/{inflow_mat}', header=None).T / 12

    deficits = np.load(f'{data_loc}/{climate_label}_deficit_array.npy')

    # Set the overall font size for plots
    plt.rcParams.update({'font.size': 18})

    # Define the RGB colors for each climate classification
    colors_req_0 = sns.color_palette("Blues", 5)
    colors_req_1 = sns.color_palette("Greens", 5)
    colors_req_2 = sns.color_palette("Purples", 5)
    colors_req_3 = sns.color_palette("YlOrBr", 5)
    colors_req_4 = sns.color_palette("Oranges", 5)
    colors_req_5 = sns.color_palette("Reds", 5)

    def get_color(idx, model_idx):
        if climate_classes[model_idx] == 0:
            return colors_req_0[idx % len(colors_req_0)]
        elif climate_classes[model_idx] == 0.5:
            return colors_req_1[idx % len(colors_req_1)]
        elif climate_classes[model_idx] == 1:
            return colors_req_2[idx % len(colors_req_2)]
        elif climate_classes[model_idx] == 1.5:
            return colors_req_3[idx % len(colors_req_3)]
        elif climate_classes[model_idx] == 2:
            return colors_req_4[idx % len(colors_req_4)]
        elif climate_classes[model_idx] == 2.5:
            return colors_req_5[idx % len(colors_req_5)]
        else:
            return 'black'

    def convert_to_annual(monthly_data):
        return monthly_data.reshape(-1, 12).sum(axis=1)

    fig1, axs1 = plt.subplots(3, 1, figsize=(25, 30))

    # Plotting annual precipitation
    for idx, model_idx in enumerate(nsim):
        annual_prep = convert_to_annual(preps.iloc[:, model_idx].values)
        axs1[0].plot(annual_prep, color=get_color(idx, model_idx), label=f'req-{climate_classes[model_idx]}-{model_idx}')
    axs1[0].set_title(f'{sim_label}_Annual Precipitation')
    axs1[0].set_xlabel('Time [years]')
    axs1[0].set_ylabel('Precipitation [mm]')
    axs1[0].set_xticks(np.arange(0, 100, 10))  # Annual data
    axs1[0].set_xticklabels(np.arange(2000, 2100, 10))
    axs1[0].legend()

    # Plotting annual inflow
    for idx, model_idx in enumerate(nsim):
        annual_inflow = convert_to_annual(inflows.iloc[:, model_idx].values)
        axs1[1].plot(annual_inflow, color=get_color(idx, model_idx), label=f'req-{climate_classes[model_idx]}-{model_idx}')
    axs1[1].set_title('Annual Inflow')
    axs1[1].set_xlabel('Time [years]')
    axs1[1].set_ylabel('Inflow [MCM]')
    axs1[1].set_xticks(np.arange(0, 100, 10))  # Annual data
    axs1[1].set_xticklabels(np.arange(2000, 2100, 10))
    axs1[1].legend()

    # Plotting annual deficit
    for idx, model_idx in enumerate(nsim):
        axs1[2].plot(deficits[model_idx], label=f'req-{climate_classes[model_idx]}-{model_idx}', color=get_color(idx, model_idx))
    axs1[2].set_title('Annual Deficit without Adding Any Capacity')
    axs1[2].set_xlabel('Time [years]')
    axs1[2].set_ylabel('Deficit [MCM]')
    axs1[2].set_xticks(np.arange(0, 100, 10))  # Annual data
    axs1[2].set_xticklabels(np.arange(2000, 2100, 10))
    axs1[2].legend()

    plt.tight_layout()
    if save_plot:
        file_name = f"{sim_label}_annual_climate_series.png"
        full_path = os.path.join(folder_path, file_name)
        plt.savefig(full_path, dpi=100)
    plt.show()


##########
def plot_indicators(nsim, normalize_min_max, policy_type, sim_label, climate_label, climate_classes, update_frequency, bayes_timing, norm_timing, label, save_plot=True):
    
    """
    Plot the indicator time series at monthly time step
    mean_indicators: 5-year moving average of inflow, 20-year moving average of precipitation(non-bayesian),bayes posterior mean precipitaion 
    std_indicators: 20-year mocing std of precipitation(non-bayesian), bayes posterior std at year 2100 

    Parameters:
    - nsim: List of model indices to plot
    - normalize_min_max: Dictionary of min and max values for normalization for bayes indicators
    - policy_type: Type of policy ('non-deficit', 'small-deficit')
    - sim_label: Label for the simulation (e.g., 'training', 'test_wet')
    - climate_label: Label for the climate data (e.g., 'test', 'test3')
    - climate_classes: List of climate classifications for each model
    - label: Label for the plot
    """
       
    folder_path = f'results/figures/{label}/{policy_type}'
    os.makedirs(folder_path, exist_ok=True)
    
    data_loc = 'data/Annual_GPR'
    inflow_indicators = pd.read_csv(f'{data_loc}/{climate_label}_inflow_indicators.csv')
    climate_means = np.load(f'{data_loc}/{climate_label}_bayes_indicators.npz')['mu']
    climate_sigmas = np.load(f'{data_loc}/{climate_label}_bayes_indicators.npz')['std']
    prep_means = pd.read_csv(f'{data_loc}/{climate_label}_prep_mean.csv',index_col = 0) #annual non-bayesian prep indicators.20-year moving average
    prep_sigmas = pd.read_csv(f'{data_loc}/{climate_label}_prep_std.csv', index_col = 0)


    # Set the overall font size for plots
    H = 100 * 12  # Number of months
    plt.rcParams.update({'font.size': 18})
    xticks = np.arange(0, 1200, 120)  # Label every 10 years
    labels = [f"{2000 + 10 * i}-01" for i in range(len(xticks))]
    
    #color settings
    # Define the RGB colors for each climate classification
    colors_req_0 = sns.color_palette("Blues", 5)
    colors_req_1 = sns.color_palette("Greens", 5)
    colors_req_2 = sns.color_palette("Purples", 5)
    colors_req_3 = sns.color_palette("YlOrBr", 5)
    colors_req_4 = sns.color_palette("Oranges", 5)
    colors_req_5 = sns.color_palette("Reds", 5)

    def get_color(idx, model_idx):
        if climate_classes[model_idx] == 0:
            return colors_req_0[idx % len(colors_req_0)]
        elif climate_classes[model_idx] == 0.5:
            return colors_req_1[idx % len(colors_req_1)]
        elif climate_classes[model_idx] == 1:
            return colors_req_2[idx % len(colors_req_2)]
        elif climate_classes[model_idx] == 1.5:
            return colors_req_3[idx % len(colors_req_3)]
        elif climate_classes[model_idx] == 2:
            return colors_req_4[idx % len(colors_req_4)]
        elif climate_classes[model_idx] == 2.5:
            return colors_req_5[idx % len(colors_req_5)]
        else:
            return 'black'
   
        

    #compute bayesian indicators
    mean_ts = np.zeros((H, len(nsim)))
    std_t2100 = np.zeros((H, len(nsim)))
    for i, model_idx in enumerate(nsim):
        for t in range(H):
            indicators = extract_climate_indicators(climate_means[model_idx, ...], climate_sigmas[model_idx, ...], t, update_frequency, normalize_min_max)
            mean_ts[t, i] = indicators[0]
            std_t2100[t, i] = indicators[2]

    # compute non-bayesian indicators
    mean_preps = np.zeros((H, len(nsim)))
    std_preps = np.zeros((H, len(nsim)))
    for i, model_idx in enumerate(nsim):
        for t in range(H):
            mean_preps[t,i] = prep_means.iloc[t//12, model_idx]
            std_preps[t,i] = prep_sigmas.iloc[t//12, model_idx]


    indicators = ['normDPS', 'bayesDPS']
    for indicator in indicators:
        fig, axs = plt.subplots(3, 1, figsize=(25, 30))

        if indicator == 'normDPS':
            titles = ['5-year Moving Average of Inflow', '20-year Moving Average of Precipitation', '20-year Moving Std of Precipitation'] 
            data = [inflow_indicators, mean_preps, std_preps]
            timing = norm_timing
        else:
            titles = ['5-year Moving Average of Inflow', 'Bayesian Posterior Mean Precipitation', 'Bayesian Posterior Std at Year 2100']
            data = [inflow_indicators, mean_ts,std_t2100]
            timing = bayes_timing
        
        for i, ax in enumerate(axs):
            for idx, model_idx in enumerate(nsim):

                color = get_color(idx, model_idx)

                if i == 0: #plot inflow indicators
                    ax.plot(data[i].iloc[:, model_idx], color=color, label=f'Req-{climate_classes[model_idx]}-{model_idx}')
                    for t in timing[idx]:
                        ax.plot(t, data[i].iloc[t, model_idx], 'o', markersize=12, color=color)

                else:
                   
                    ax.plot(data[i][:, idx], color=color, label=f'Req-{climate_classes[model_idx]}-{model_idx}')
                    for t in timing[idx]:
                        ax.plot(t, data[i][t, idx], 'o', markersize=12, color=color)

            ax.set_title(titles[i])
            ax.set_xlabel('Time')
            ax.set_ylabel('Normalized Indicator')
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels)
            ax.legend()

        plt.tight_layout()
        if save_plot:
            file_name = f"{indicator}_indicators_timeseries.png"
            full_path = os.path.join(folder_path, file_name)
            plt.savefig(full_path, dpi=100)
        plt.show()


                   




def plot_capacity(nsim, policy_type, sim_label, bayes_capacity_series,norm_capacity_series,  climate_classes, label, save_plot = True):
    
    folder_path = f'results/figures/{label}/{policy_type}'
    os.makedirs(folder_path, exist_ok=True)

    H = 100 * 12  # Number of months
    # Set the overall font size for plots
    plt.rcParams.update({'font.size': 18})


    # Define the RGB colors for each climate classification
    colors_req_0 = sns.color_palette("Blues", 5)
    colors_req_1 = sns.color_palette("Greens", 5)
    colors_req_2 = sns.color_palette("Purples", 5)
    colors_req_3 = sns.color_palette("YlOrBr", 5)
    colors_req_4 = sns.color_palette("Oranges", 5)
    colors_req_5 = sns.color_palette("Reds", 5)

    def get_color(idx, model_idx):
        if climate_classes[model_idx] == 0:
            return colors_req_0[idx % len(colors_req_0)]
        elif climate_classes[model_idx] == 0.5:
            return colors_req_1[idx % len(colors_req_1)]
        elif climate_classes[model_idx] == 1:
            return colors_req_2[idx % len(colors_req_2)]
        elif climate_classes[model_idx] == 1.5:
            return colors_req_3[idx % len(colors_req_3)]
        elif climate_classes[model_idx] == 2:
            return colors_req_4[idx % len(colors_req_4)]
        elif climate_classes[model_idx] == 2.5:
            return colors_req_5[idx % len(colors_req_5)]
        else:
            return 'black'
        
    

    xticks = np.arange(0, 1200, 120)  # Label every 10 years
    labels = [f"{2000 + 10 * i}-01" for i in range(len(xticks))]

    fig, axs = plt.subplots(2, 1, figsize=(25, 20))
    # plot norm_policy capacity expansion time series
    for idx, model_idx in enumerate(nsim):
        axs[0].plot(norm_capacity_series[idx], lw = 2, color=get_color(idx, model_idx), label=f'req-{climate_classes[model_idx]}-{model_idx}')
    axs[0].set_title(f'Norm {policy_type} Capacity Expansion Series under {sim_label} climates')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Capacity [MCM]')
    axs[0].legend()

    # plot bayes_policy capacity expansion time series
    for idx, model_idx in enumerate(nsim):
        axs[1].plot(bayes_capacity_series[idx], lw =2, color=get_color(idx, model_idx), label=f'req-{climate_classes[model_idx]}-{model_idx}')
    axs[1].set_title(f'Bayes {policy_type} Capacity Expansion Series under {sim_label} climates')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Capacity [MCM]')
    axs[1].legend()

    plt.tight_layout()
    if save_plot:
        file_name = "capacity_expansion_series.png"
        full_path = os.path.join(folder_path, file_name)
        plt.savefig(full_path, dpi=100)
    plt.show()






def plot_climates_series_notused(nsim, sim_label,  climate_label,policy_type,  climate_classes, update_frequency, bayes_timing, norm_timing, label, save_plot=True):
    
    '''
    sim_label: 'training','test' etc
    climate_label: 'test', 'test3' etc
    policy_type: 'non-deficit','small-deficit'
    '''
    folder_path = f'results/figures/{label}/climate_time_series'
    os.makedirs(folder_path, exist_ok=True)

    data_loc = 'data/Annual_GPR'
    xticks = np.arange(0, 1200, 120)   # label every 10 year
    labels = [f"{2000 + 10 * i}-01" for i in range(len(xticks))]
    H = 100 * 12  # Number of months

    num_models = len(nsim)
    
    prep_mat = f'{climate_label}_month_prep.csv'
    preps = pd.read_csv(f'{data_loc}/{prep_mat}', header=None, skiprows=1)

    inflow_mat = f'{climate_label}_runoff.csv'
    inflows = pd.read_csv(f'{data_loc}/{inflow_mat}', header=None).T / 12

    deficits = np.load(f'{data_loc}/{climate_label}_deficit_array.npy')

    inflow_indicators_name = f'{climate_label}_inflow_indicators.csv'
    inflow_indicators = pd.read_csv(f'{data_loc}/{inflow_indicators_name}')

    bayes_mat = f'{climate_label}_bayes_indicators.npz'
    climate_means = np.load(f'{data_loc}/{bayes_mat}')['mu']
    climate_sigmas = np.load(f'{data_loc}/{bayes_mat}')['std']
    
    # Set the overall font size for plots
    plt.rcParams.update({'font.size': 18})

    # Pre-allocate numpy arrays for each set of indicators
    mean_ts = np.zeros((H, num_models))
    mean_t25s = np.zeros((H, num_models))
    mean_t2100 = np.zeros((H, num_models))
    std_t2100 = np.zeros((H, num_models))

    for i, model_idx in enumerate(nsim):
        for t in range(H):
            # Extract indicators
            indicators = extract_climate_indicators(climate_means[model_idx, ...], climate_sigmas[model_idx, ...], update_frequency, t)
            mean_ts[t, i] = indicators[0]
            mean_t25s[t, i] = indicators[1]
            mean_t2100[t, i] = indicators[2]
            std_t2100[t, i] = indicators[3]

    # Define the RGB colors for each climate classification
    #colors_req_0 = [(0.1, 0.2, 0.5), (0.2, 0.4, 0.7), (0.3, 0.6, 0.9), (0.4, 0.8, 1.0), (0.5, 1.0, 1.0)]
    #colors_req_1 = [(0.1, 0.5, 0.1), (0.2, 0.7, 0.2), (0.3, 0.9, 0.3), (0.4, 1.0, 0.4), (0.5, 1.0, 0.5)]
    #colors_req_2 = [(0.5, 0.1, 0.1), (0.7, 0.2, 0.2), (0.9, 0.3, 0.3), (1.0, 0.4, 0.4), (1.0, 0.5, 0.5)]
    #colors_req_3 = [(0.1, 0.1, 0.6), (0.2, 0.2, 0.7), (0.3, 0.3, 0.8), (0.4, 0.4, 0.9), (0.5, 0.5, 1.0)]
    #colors_req_4 = [(0.6, 0.1, 0.1), (0.7, 0.2, 0.2), (0.8, 0.3, 0.3), (0.9, 0.4, 0.4), (1.0, 0.5, 0.5)]
    colors_req_0 = sns.color_palette("Blues", 5)
    colors_req_1 = sns.color_palette("Greens", 5)
    colors_req_2 = sns.color_palette("Purples", 5)
    colors_req_3 = sns.color_palette("Oranges", 5)
    colors_req_4 = sns.color_palette("Reds", 5)

    ########## need to change the value of cliamte classess for different cliamtes sets
    def get_color(idx, model_idx):
        if climate_classes[model_idx] == 0:
            return colors_req_0[idx % len(colors_req_0)]
        elif climate_classes[model_idx] == 0.5:
            return colors_req_1[idx % len(colors_req_1)]
        elif climate_classes[model_idx] == 1:
            return colors_req_2[idx % len(colors_req_2)]
        elif climate_classes[model_idx] == 2:
            return colors_req_3[idx % len(colors_req_3)]
        elif climate_classes[model_idx] == 4:
            return colors_req_4[idx % len(colors_req_4)]
        else:
            return 'black'

    # Create figure and axes

    # Convert monthly data to annual data
    def convert_to_annual(monthly_data):
        return monthly_data.reshape(-1, 12).sum(axis=1)

    # Create the figure for annual data
    fig1, axs1 = plt.subplots(3, 1, figsize=(25, 30))

    # Plotting annual precipitation
    for idx, model_idx in enumerate(nsim):
        annual_prep = convert_to_annual(preps.iloc[:, model_idx].values)
        axs1[0].plot(annual_prep, color=get_color(idx, model_idx), label= f'req-{climate_classes[model_idx]}-{model_idx}')
    axs1[0].set_title(f'{sim_label}_Annual Precipitation')
    axs1[0].set_xlabel('Time [years]')
    axs1[0].set_ylabel('Precipitation [mm]')
    axs1[0].set_xticks(np.arange(0, 100, 10))  # Annual data
    axs1[0].set_xticklabels(np.arange(2000, 2100, 10))
    axs1[0].legend()

    # Plotting annual inflow
    for idx, model_idx in enumerate(nsim):
        annual_inflow = convert_to_annual(inflows.iloc[:, model_idx].values)
        axs1[1].plot(annual_inflow, color=get_color(idx, model_idx), label= f'req-{climate_classes[model_idx]}-{model_idx}')
    axs1[1].set_title('Annual Inflow')
    axs1[1].set_xlabel('Time [years]')
    axs1[1].set_ylabel('Inflow [MCM]')
    axs1[1].set_xticks(np.arange(0, 100, 10))  # Annual data
    axs1[1].set_xticklabels(np.arange(2000, 2100, 10))
    axs1[1].legend()

    # Plotting annual deficit
    for idx, model_idx in enumerate(nsim):
        axs1[2].plot(deficits[model_idx], label = f'req-{climate_classes[model_idx]}-{model_idx}', color=get_color(idx, model_idx))
    axs1[2].set_title('Annual Deficit withou adding any capacity')
    axs1[2].set_xlabel('Time [years]')
    axs1[2].set_ylabel('Deficit [MCM]')
    axs1[2].set_xticks(np.arange(0, 100, 10))  # Annual data
    axs1[2].set_xticklabels(np.arange(2000, 2100, 10))
    axs1[2].legend()

    plt.tight_layout()
    if save_plot:
        file_name = f"{sim_label}_annual_climate_series.png"
        full_path = os.path.join(folder_path, file_name)
        plt.savefig(full_path, dpi=100)
    plt.show()

    # Create the figure for monthly  indicators data

    if sim_label == 'training':
        fig, axs = plt.subplots(2, 1, figsize=(25, 20))

        # Plotting inflow indicators
        for idx, model_idx in enumerate(nsim):
            axs[0].plot(inflow_indicators.iloc[:, model_idx], color=get_color(idx, model_idx), label= f'req-{climate_classes[model_idx]}-{model_idx}')
            # Overlay timing data
            for t in norm_timing[idx]:
                axs[0].plot(t, inflow_indicators.iloc[t, model_idx], 'o', markersize=12, color=get_color(idx, model_idx))
        axs[0].set_title(f'{label}_Inflow Indicators with {policy_type} timing')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Normalized Inflow Indicators')
        axs[0].set_xticks(xticks)
        axs[0].set_xticklabels(labels)
        axs[0].legend()

        # Plotting mean_ts indicators
        for idx, model_idx in enumerate(nsim):
            axs[1].plot(mean_ts[:, idx], color=get_color(idx, model_idx), label = f'req-{climate_classes[model_idx]}-{model_idx}')
            # Overlay timing data
            for t in bayes_timing[idx]:
                axs[1].plot(t, mean_ts[t, idx], 'o', markersize=12, color=get_color(idx, model_idx))
        axs[1].set_title(f'Bayes Posterior Climate Mean at Time t with {policy_type} timing')
        axs[1].set_ylabel('Normalized Posterior Mean')
        axs[1].set_xlabel('Time')
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(labels)
        axs[1].legend()

        plt.tight_layout()
        if save_plot:
            file_name = f"indicators_with{policy_type} timing.png"
            full_path = os.path.join(folder_path, file_name)
            plt.savefig(full_path, dpi=100)
        plt.show()


def plot_heatmap_policy(policy_params, policy_type, policy_label,plot_discret, config, K, label, save_plot = True):
    '''
    policy_params: the policy parameters
    policy_type: 'non-deficit','small-deficit'
    policy_label: 'bayes', 'norm
    plot_discret: True or False. if true, plot the discretized capacity, else, plot direct outout from rbf policy
    
    This function generates heatmaps to visualize the policy.
    For BayesDPS, it plots two heatmaps side by side for std = 0.8 and std = 0.2, with mu on the x-axis and obs on the y-axis.
    #For normDPS, it plots a 1D heatmap with obs on the x-axis.
    The color in the heatmaps indicates the capacity value.
    If a flexible pipe is installed, it adds '//' to the corresponding grid.
    
    
    '''
    folder_path = f'results/figures/{label}/{policy_type}'
    os.makedirs(folder_path, exist_ok=True)
    # Set the overall font size for plots
    plt.rcParams.update({'font.size': 18})  # Adjust the base font size

    flex = False
    # Add 0 in front of config.sizes
    discrete_sizes = [0] + list(config.sizes)


    if policy_label == 'bayes' or policy_label == 'norm':
        N,M = 5,3
        fig, axes = plt.subplots(1, 2, figsize=(25, 10))
   
        for idx, std in enumerate([0.9, 0.4]):
            heatmap_data = np.zeros((101, 101))  # Create a 101x101 grid for the heatmap
            flex_data = np.zeros((101, 101), dtype=bool)  # Create a boolean grid for flexible pipe indicator
        
            for obs in range(101):
                    for mu in range(101):
                        indicator = [obs / 100, mu / 100, std]
                        capacity, flex = policy2capacity2(policy_params, indicator, config, N, M, K, plot_discret)
                        heatmap_data[obs, mu] = capacity
                        flex_data[obs, mu] = flex
            map = 'viridis'
            if plot_discret:
                cmap = plt.cm.get_cmap('viridis', len(discrete_sizes))
                norm = plt.Normalize(vmin=0, vmax=max(discrete_sizes))
                cbar_kws = {
                    'label': 'Discretized Capacity',
                    'ticks': discrete_sizes,
                    'extend': 'neither'
                }
            else:
                cmap = 'viridis'
                norm = None
                cbar_kws = {'label': 'Capacity'}
            
            #print(f'bayes flex_{std}:', flex_data)

            heatmap_data = np.flipud(heatmap_data)
            flex_data = np.flipud(flex_data)

            sns.heatmap(heatmap_data, cmap=cmap, norm=norm,  cbar_kws=cbar_kws, ax=axes[idx])
            axes[idx].set_title(f'{policy_label} policy with std = {std}, discret = {plot_discret}')
            if policy_label == 'bayes':
                axes[idx].set_xlabel('bayesian posterior mean precipitation')
            if policy_label == 'norm':
                axes[idx].set_xlabel('none-bayesian 20-year moving average of precipitation')

            axes[idx].set_ylabel('5-year moving average of inflow')
            axes[idx].set_aspect('equal')  # Set the aspect ratio to equal
            axes[idx].set_xticks(np.linspace(0, 100, 11))
            axes[idx].set_xticklabels(np.round(np.linspace(0, 1, 11), 2))
            axes[idx].set_yticks(np.linspace(100, 0, 11))
            axes[idx].set_yticklabels(np.round(np.linspace(0, 1, 11), 2))
    
    
            for obs in range(101):
                for mu in range(101):
                    if flex_data[obs, mu]:
                        axes[idx].text(mu + 0.5, obs + 0.5, '.', ha='center', va='center', color='white')
                        
    
       
        # Add 'flexible /' at the bottom
        fig.text(0.5, 0.06, "'/' : flexible installed", ha='center', va='bottom', fontsize=12)

        if save_plot:
            plt.savefig(f'{folder_path}/{policy_label}_policy heatmap_discret = {plot_discret}.png')
        plt.show()

    
    elif policy_label == '1Dnorm':
        N,M = 3,1
        heatmap_data = np.zeros((101,1))  # Create a 101 * 1 grid for the heatmap
        flex_data = np.zeros((101,1), dtype=bool)  # Create a boolean grid for flexible pipe indicator
        #The heatmap will display a vertical strip with 101 rows and 1 column.

        for obs in range(101):
            indicator = [obs / 100]
            capacity, flex = policy2capacity(policy_params, indicator, config, N, M, K, plot_discret)
            heatmap_data[obs,0] = capacity
            flex_data[obs,0] = flex
        
        fig,ax = plt.subplots(figsize=(12, 10))
        if plot_discret:
            cmap = plt.cm.get_cmap('viridis', len(discrete_sizes))
            norm = plt.Normalize(vmin=0, vmax=max(discrete_sizes))
            cbar_kws = {
                'label': 'Discretized Capacity',
                'ticks': discrete_sizes,
                'extend': 'neither'
            }
        else:
            cmap = 'viridis'
            norm = None
            cbar_kws = {'label': 'Capacity'}
        
        heatmap_data = np.flipud(heatmap_data)
        flex_data = np.flipud(flex_data)

        sns.heatmap(heatmap_data, cmap=cmap, norm=norm, cbar_kws=cbar_kws)
        plt.title(f'Norm policy (discret = {plot_discret})')
        plt.ylabel('obs')
        plt.xlabel('')
        plt.xticks([])
        plt.yticks(np.linspace(100, 0, 11), np.round(np.linspace(0,1, 11), 2))
        
        for obs in range(101):
            for x in np.linspace(0, 1, num=101):  # Adjust num=11 for how many symbols you want
                if flex_data[obs, 0]:
                    ax.text(x, obs+0.5, '/', horizontalalignment='center', verticalalignment='center', color='white')
            
        # Add the legend for the '/' symbol
        ax.text(0.5, 0.01, "'/' : flexible installed", ha='center', va='bottom', fontsize=12, transform=fig.transFigure)

        if save_plot:
            plt.savefig(f'{folder_path}/norm_policy heatmap_discret = {plot_discret}.png')
        plt.show()
    

def policy2capacity(P, indicator, config, N, M, K, plot_discret):

    """
    1. Determine the capacity output of a Neural network policy based on given indicators.
    2. Either directly return the capacity, or map this capacity to a discrete set of sizes and determine whether a flexible pipe is being installed.
    """
   
    nn = set_param(P, M, N, K)  # set the parameters for the neural network
    #obtain output from rbf function
    capacity = get_output(indicator, nn)[0]
    
    if plot_discret == False:
        flex=False
        return capacity, flex
    else: 
        flex = False   # if flexible pipe being installed?
        n_options = len(config.sizes)
        seg = 1 / (n_options * 2 + 1) if config.option_type == 'both' else 1 / (n_options + 1)
        if config.option_type == 'flexible':
            flex = True
        
        size_array = np.concatenate((np.array([0]), config.sizes, config.sizes))
        capacity_idx = capacity // seg # the index of the capacity in the size_array
        
        if capacity_idx >= len(size_array):
            capacity_idx = len(size_array) - 1   #potential out of bounds error handling

        capa_installed = size_array[int(capacity_idx)]
        
        if config.option_type =='both' and capacity_idx > n_options:
            flex = True
        return capa_installed, flex

def policy2capacity2(P, indicator, config, N, M, K, plot_discret):
    """
    Use softmax in the output layer to determine the capacity output of a Neural network policy based on given indicators.
    1. Determine the capacity output of a Neural network policy based on given indicators.
    2. Either directly return the capacity, or map this capacity to a discrete set of sizes and determine whether a flexible pipe is being installed.
    
    """
    nn = set_param(P, M, N, K)  # set the parameters for the neural network
    prob = get_output(indicator, nn)

    if plot_discret == False:
        flex = False
        return prob[1], flex    #return the prob of installing a small pipe
    else:
        flex = False   # if flexible pipe being installed?

        action_idx = np.argmax(prob)
        n_sizes = len(config.sizes)

        if action_idx ==0:
            return 0, flex
        elif config.option_type == 'flexible':
            flex = True
            return config.sizes[action_idx-1], flex
        elif config.option_type == 'both':
            if action_idx <= n_sizes:
                return config.sizes[action_idx-1], flex
            else:
                flex = True
                return config.sizes[action_idx - n_sizes - 1], flex
        elif config.option_type == 'fixed':
            return config.sizes[action_idx-1], flex
    




            
def plot_trajectories(param, opt_par, config, s, config_name, label,save_plot = True):  #decide simulate under which inflow
    
    folder_path = f'results/figures/{label}'
    os.makedirs(folder_path, exist_ok=True)

    sns.set_style('darkgrid')

    sim = Pipe_sim(opt_par, config)
    traj = sim.simulation(param, s)
    
    # read in precipitation
    #data_loc = '/Users/mofanz/Documents/OneDrive - Stanford/research/4. 2024/project/Bayesian DPS/coding/synthetic generator/results'
    if config.from_5yr ==True:
        data_loc = 'data/5yr_GPR'
        xticks = np.arange(0, 1200, 300)   # label every 25 year
        labels = [f"{2000 + 25 * i}-01" for i in range(len(xticks))]
        #labels_2 = [f"{2050 + 25 * i}-01" for i in range(len(xticks))]
        labels_2 = [f"{2000 + 25* i}-01\n{2050 + 25 * i}-01" for i in range(len(xticks))]
    else:
        data_loc = 'data/annual_GPR'
        xticks = np.arange(0, 1200, 240)
        labels = [f"{2000 + 20 * i}-01" for i in range(len(xticks))]
        #labels_2 = [f"{2050 + 20 * i}-01" for i in range(len(xticks))]
        labels_2 = [f"{2000 + 20 * i}-01\n{2050 + 20 * i}-01" for i in range(len(xticks))]

    prep_mat = 'test_month_prep.csv'
    preps = pd.read_csv(f'{data_loc}/{prep_mat}', header = None)
    
    # Set the overall font size for plots
    plt.rcParams.update({'font.size': 12})  # Adjust the base font size

    ################################## First figure
    #Prec
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(preps.iloc[:,s].values)
    axs[0].set_title('Precipitation')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Monthly Precipitation(mm/month)')
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(labels)

    # Inflow
    axs[1].plot(traj.inflow)
    axs[1].set_title('Inflow')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Monthly inflow (MCM/month)')
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(labels)

    # Storage
    axs[2].plot(traj.storage)
    axs[2].set_title('Reservoir Storage')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Storage [MCM/month]')
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(labels)


    plt.tight_layout()

    file_name = f"climate{s}_{config_name}_prep_inflow_storage.png"
    full_path = os.path.join(folder_path, file_name)

    if save_plot: 
        plt.savefig(full_path, dpi=100)

        #plt.savefig(f'results/figures/{config_name}_climate{s}prep_inflow_storage.png', dpi=100)
        plt.show()

    ############## ###Second Figure: Release , Deficit and capacity
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # Adjust the figure size as needed
    # Release
    axs[0].plot(traj.release)
    axs[0].set_title('Reservoir Release')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Release [MCM/month]')
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(labels)

    # Deficit
    axs[1].plot(traj.deficit)
    axs[1].set_title('Deficit')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Deficit [MCM/month]')
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(labels)

    # Capacity
    if traj.fix_built:
        axs[2].plot(traj.capacity,'ro', label='Fixed Capacity')
    elif traj.flex_built:
        axs[2].plot(traj.capacity,'bo', label='Flex Capacity')
    else:
        axs[2].plot(traj.capacity,'go', label='No Capacity')
    axs[2].set_title('Installed Capacity')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Capacity [MCM]')
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(labels)
    axs[2].legend()
    
    plt.tight_layout()
    
    if save_plot: 

        file_name = f"climate{s}_{config_name}_release_deficit_capacity.png"
        full_path = os.path.join(folder_path, file_name)
        plt.savefig(full_path, dpi=100)

        #plt.savefig(f'results/figures/{config_name}_climate{s}release_deficit_capacity.png', dpi=100)
        plt.show()

    print(f'the total cost of {config_name} is: Jplan: {traj.Jplan}; Jdeficit : {traj.Jdef}')

    ############ Third figure, plot cliamte indicators

    # Extract climate means and stds
    climate_means = config.climate_means[s, ...]  
    climate_stds = config.climate_sigmas[s, ...] 
    # Initialize lists to store the time series data for climate indicators
    norm_mean_ts = []
    norm_mean_2100 = []
    norm_std_2100 = []


    # Calculate indicators
    H = config.H
    for t in range(H):  # Assuming 1200 timesteps
        indicators = sim.extract_climate_indicators(climate_means, climate_stds, t)
        norm_mean_ts.append(indicators[0])
        #norm_mean_2100.append(indicators[1])
        norm_std_2100.append(indicators[1])
    
    # Plotting

    # Create figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))

    #dates = np.arange('2000-01', '2100-01', dtype='datetime64[M]')  #generate monthly timestamps starting from Jan 2000 to Dec 2099

    dates = np.arange(0, 1200)


    #3.1 Plotting mean(t) and mean(t+50) in the first subplot
    axs[0].plot(dates, norm_mean_ts, label='Mean at Time t', color='blue', linestyle='--')
    #axs[0].plot(dates, norm_mean_2100, label='Mean at Time 2100 year', color='red')
    #axs[0].plot(dates, norm_mean_t_50s, label='Mean at Time t+50year', color='red')
    axs[0].set_title('Climate Mean at Time t')
    axs[0].set_ylabel('Normalized Posterior Mean')
    axs[0].legend()

    # Handling the x-axis labels
    # Set the x-ticks for every 20 years, assuming each tick represents a month

    # Custom labels for every 20 years, starting from 2000 and 2050 respectively
    #labels_20_year_cycle = [f"{2000 + 20 * i}-01\n{2050 + 20 * i}-01" for i in range(len(xticks))]
    
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(labels)
    # Configure ticks to show the expected dual-year labeling
    plt.setp(axs[0].get_xticklabels(), rotation=0, ha='center')

 
    #3.2  Plotting std(t2100) on the second subplot
    axs[1].plot(dates, norm_std_2100, label='Std at Time 2100 year', color='green')
    axs[1].set_title('Standard Deviation at Time 2100 year')
    axs[1].set_ylabel('Normalized Posterior Std')
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(labels)

    #3.3  Plotting inflow indicators on the third subplot
    indicator = config.inflow_indicators.iloc[:,s]
    axs[2].plot(dates, indicator)
    axs[2].set_title('Inflow indicators: rolling mean of previous 5 year')
    axs[2].set_xlabel('Year-Month')
    axs[2].set_ylabel('Normalized inflow indicators')
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(labels)


    plt.tight_layout()
    file_name = f"climate{s}_{config_name}_climate_indicators.png"
    full_path = os.path.join(folder_path, file_name)
    if save_plot: 
        plt.savefig(full_path, dpi=100)
        #plt.savefig(f'results/figures/{config_name}_climate{s}_climate_indicators.png', dpi=100)
        plt.show()

def evaluate_policy(param, opt_par, config, nsim):  #simulate under nsim, and return the Jplan and Jdeficit
    
    Jplan = []
    Jdeficit = []
    flex_built = []
    installed_capacity = []
    timing =  []
    capacity_series = []

    #simulate policies and save results
    sim = Pipe_sim(opt_par, config)
    for s in nsim:
        traj = sim.simulation(param, s)
        Jplan.append(traj.Jplan)
        Jdeficit.append(traj.Jdef)
        flex_built.append(traj.flex_built)
        installed_capacity.append(traj.capacity[-1])   # total capacity installed
        capacity_series.append(traj.capacity)   # cacacity installed over time time series 
        timing.append(traj.timing)

    
    result = {'Jplan': Jplan, 'Jdeficit': Jdeficit, 'flex_built': flex_built, 'installed_capacity': installed_capacity, 'capacity_series':capacity_series, 'timing': timing}
    return result

def save_timing(bayes_result, norm_result,  save_json=True):
    """
    Save timing data from bayes_result and norm_result to JSON and/or CSV files.

    Parameters:
    - bayes_result: Dictionary containing the Bayes timing data.
    - norm_result: Dictionary containing the Norm timing data.
    - save_json: Boolean to save data as JSON. Default is True.
    """

    file_path = 'data/annual_GPR'
    bayes_timing_data = bayes_result['timing']
    norm_timing_data = norm_result['timing']

    if save_json:
        with open(os.path.join(file_path, 'bayes_timing_data.json'), 'w') as json_file:
            json.dump(bayes_timing_data, json_file)

        with open(os.path.join(file_path, 'norm_timing_data.json'), 'w') as json_file:
            json.dump(norm_timing_data, json_file)

        print("Bayes and Norm timing data saved as JSON files")

def plot_policy_compare(bayes_result, norm_result, policy_label, label_tag, trend_label, nsims, plot_bar_flag = True,save_plot_flag = True):
   
    '''
    Make bar plot under training sets to compare the Jplan and Jdeficit between Bayes and Norm policy
    policy_label: 'non-deficit','small-deficit'
    label_tag: e.g. 071601
    trend_label: 'training','test' etc
    nsims: list of indices of climates
    '''

    folder_path = f'results/figures/{label_tag}/{policy_label}'
    #print(folder_path)
    os.makedirs(folder_path, exist_ok=True)

    if plot_bar_flag:
        bayes_Jplan = bayes_result['Jplan']
        bayes_Jdeficit = bayes_result['Jdeficit']
        bayes_flex_built = bayes_result['flex_built']

        norm_Jplan = norm_result['Jplan']
        norm_Jdeficit = norm_result['Jdeficit']
        norm_flex_built = norm_result['flex_built']

        n_scenarios = len(bayes_Jplan)
        indices = np.arange(n_scenarios)
        width = 0.35
        def add_value_labels(ax, bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)

        # Plotting Jplan
        fig, ax = plt.subplots(figsize=(15, 8))
        bar1 = ax.bar(indices - width/2, bayes_Jplan, width, label='BayesDPS')
        bar2 = ax.bar(indices + width/2, norm_Jplan, width, label='NormDPS')

        # Apply hatch pattern for scenarios where flexible infrastructure was built
        for i in range(n_scenarios):
            
            if bayes_flex_built[i]:
                bar1[i].set_hatch('\\')
            if norm_flex_built[i]:
                bar2[i].set_hatch('\\')

        add_value_labels(ax, bar1)
        add_value_labels(ax, bar2)

        ax.set_xlabel('Scenario')
        ax.set_ylabel('Cumulative Planning Cost (Jplan)')
        ax.set_title(f'{trend_label}_{policy_label}_Comparison of Cumulative Planning Cost (Jplan) for BayesDPS and NormDPS')
        ax.set_xticks(indices)
        ax.set_xticklabels([f' {i}' for i in nsims])
        ax.legend()
        
        file_name = f"barplot_planning_cost_comparison_climate_{trend_label}.png"
        full_path = os.path.join(folder_path, file_name)
        plt.tight_layout()
        if save_plot_flag: 
            plt.savefig(full_path, dpi=100)
        plt.show()


        # Plotting Jdeficit
        fig, ax = plt.subplots(figsize=(15, 8))
        bar1 = ax.bar(indices - width/2, bayes_Jdeficit, width, label='BayesDPS')
        bar2 = ax.bar(indices + width/2, norm_Jdeficit, width, label='NormDPS')

        add_value_labels(ax, bar1)
        add_value_labels(ax, bar2)

        ax.set_xlabel('Scenario')
        ax.set_ylabel('Cumulative Water Shortage Cost (Jdeficit)')
        ax.set_title(f'{trend_label}_{policy_label}_Comparison of Cumulative Water Shortage Cost (Jdeficit) for BayesDPS and NormDPS')
        ax.set_xticks(indices)
        ax.set_xticklabels([f'{i}' for i in nsims])
        ax.legend()
        file_name = f"barplot_Jdeficit_comparison_climate_{trend_label}.png"
        full_path = os.path.join(folder_path, file_name)
        plt.tight_layout()
        if save_plot_flag: 
            plt.savefig(full_path, dpi=100)
        plt.show()

def plot_planning_result_notused(Bayes_result, Norm_result, label,result_type, require_0, require_1, require_2, save_plot = True, box=True):

    """
    Create a box plot comparing key results, e.g.g,  planning costs for Bayes and Norm policies across different climate sets.

    Parameters:
    - Bayes_result: List of planning costs for the Bayes policy across 60 scenarios. 
    - Norm_result: List of planning costs for the Norm policy across 60 scenarios.
    - require_0: List of indices characterizing climates that require 0 capacity.
    - require_1: List of indices characterizing climates that require 1 capacity.
    - require_2: List of indices characterizing climates that require 2 capacities.
    - box: whether to plot box or violion plot
    """

    folder_path = f'results/figures/{label}'
    os.makedirs(folder_path, exist_ok=True)


    #Define climate sets
    climate_sets = [require_0, require_1, require_2]
    climate_labels = ['Req 0', 'Req 1', 'Req 2']

    # Extract planning costs for each climate set
    data = []
    labels = []
    positions = []
    colors = []
    for idx, climate_set in enumerate(climate_sets):
        data.append([Bayes_result[i] for i in climate_set])
        labels.append(f'Bayes {climate_labels[idx]}')
        colors.append('blue')
        data.append([Norm_result[i] for i in climate_set])
        labels.append(f'Norm {climate_labels[idx]}')
        colors.append('pink')
    
    # Create the box plot
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, label in enumerate(climate_labels):
        positions.extend([i * 2, i * 2 + 0.5])
        labels.extend([f'Bayes {label}', f'Norm {label}'])
    
    if box:
    # Create box plot
        for i, (box_data, color) in enumerate(zip(data, colors * len(climate_labels))):
            bp = ax.boxplot(box_data, positions=[positions[i]], widths=0.4, patch_artist=True)
            for element in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
                plt.setp(bp[element], color=color)
            plt.setp(bp['boxes'], facecolor=color, alpha=0.5)
            
            # Add points to represent the costs
            for point in box_data:
                ax.scatter(positions[i], point, color=color, alpha=0.7)

 
    else:
        # Create violin plot
        sns.violinplot(data=data, ax=ax, inner=None, palette=colors)
        for i in range(len(data)):
            sns.swarmplot(data=[data[i]], ax=ax, color='k', alpha=0.5)
   
   # ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Customize the plot
    ax.set_title(f'Comparison of {result_type}for Bayes and Norm Policies Across Different Climate Sets', fontsize = 16)
    ax.set_xlabel('Climate Sets', fontsize = 14)
    ax.set_ylabel(f'{result_type}', fontsize = 14)
    ax.set_xticks([0.25, 2.25, 4.25])
    ax.set_xticklabels(climate_labels, fontsize = 12)
   
    # Increase font size of x-ticks and y-ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add legend
    blue_patch = plt.Line2D([0], [0], color='blue', lw=4, label='BayesDPS')
    pink_patch = plt.Line2D([0], [0], color='pink', lw=4, label='NormDPS')
    ax.legend(handles=[blue_patch, pink_patch], fontsize=12)
    
    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()

    file_name = f"{result_type}_comparison across cliamtes.png"
    full_path = os.path.join(folder_path, file_name)
    if save_plot: 
        plt.savefig(full_path, dpi=100)
    plt.show()

def plot_multiple_results(bayes_result, norm_result, nsim, policy_type, label, result_types, climate_classes, save_plot=True,plot_type = 'box'):
    """
    Create a box plot comparing multiple key results for Bayes and Norm policies across different climate sets.

    Parameters:
    - bayes_result: Dictionary of results for the Bayes policy across scenarios.
    - norm_result: Dictionary of results for the Norm policy across scenarios.
    - nsim: list of indices of simulation climates
    - policy_type: 'non-deficit','small-deficit'
    - label: e.g. 071601, label for ouput directory.
    - result_types: List of result types to plot (e.g., ['installed_capacity', 'Jplan', 'Jdeficit']).
    - climate_classes: Array of climate classifications e.g.(0, 1, 2, 3, 4).
    - save_plot: Whether to save the plot.
    - box: Whether to plot box or violin plot.
    """

    folder_path = f'results/figures/{label}/{policy_type}'
    os.makedirs(folder_path, exist_ok=True)

    plt.rcParams.update({'font.size': 16})  # Adjust the base font size

    fig, axs = plt.subplots(len(result_types), 1, figsize=(12, 8 * len(result_types)))
    colors = {'Bayes': 'blue', 'Norm': 'pink'}

    for idx, result_type in enumerate(result_types):
        ax = axs[idx]
        
        # Prepare data for plotting
        data = {
            'climates': [],
            'result': [],
            'policy': []
        }

        # Prepare data based on unique climate classes
        #unique_climate_classes = np.unique(climate_classes)

        #unique_climate_classes = [0, 0.5, 1, 2]
        #unique_climate_classes = [0.5, 1, 1.5, 2, 2.5]
        unique_climate_classes = np.unique([climate_classes[i] for i in nsim])

        for climate_class in unique_climate_classes:

            # Get indices where climate class matches the desired class within nsim
            matched_indices = [idx for idx, i in enumerate(nsim) if climate_classes[i] == climate_class]

            # Fetch the corresponding bayes and norm results using these indices
            bayes_data = [bayes_result[result_type][idx] for idx in matched_indices]
            norm_data = [norm_result[result_type][idx] for idx in matched_indices]

            data['climates'].extend([climate_class] * len(bayes_data))
            data['result'].extend(bayes_data)
            data['policy'].extend(['Bayes'] * len(bayes_data))

            data['climates'].extend([climate_class] * len(norm_data))
            data['result'].extend(norm_data)
            data['policy'].extend(['Norm'] * len(norm_data))
        
        # Add "All Climates" data
        bayes_all_data = bayes_result[result_type]
        norm_all_data = norm_result[result_type]
        all_climate_class = 'All Climates'
        
        data['climates'].extend([all_climate_class] * len(bayes_all_data))
        data['result'].extend(bayes_all_data)
        data['policy'].extend(['Bayes'] * len(bayes_all_data))

        data['climates'].extend([all_climate_class] * len(norm_all_data))
        data['result'].extend(norm_all_data)
        data['policy'].extend(['Norm'] * len(norm_all_data))
        
        # Convert to DataFrame
        df = pd.DataFrame(data)


        if plot_type == 'box':
            # Create box plot
            boxplot = sns.boxplot(data=df, x='climates', y='result', hue='policy', ax=ax, palette=colors, width = 0.6, fliersize =0)
            for patch in boxplot.patches:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.3))
        
        else:
            # Create violin plot
            violinplot = sns.violinplot(data=df, x='climates', y='result', hue='policy', split=True, inner='quart', ax=ax, palette= colors)
            # Adjust alpha for violin plot
            for patch in violinplot.collections:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, 0.3))

        # add individual points
        sns.stripplot(x='climates', y='result', data=df, hue='policy', palette=colors, jitter=True, dodge=True, ax=ax, alpha=0.9, size=8)
        
   
        # Customize the plot
        ax.set_title(f'{label}_ {result_type} for Bayes and Norm {policy_type} Policies Across Test Climate Sets')
        ax.set_xlabel('Climate Sets')
        
           # Update xticks and xticklabels dynamically
        all_labels = [f'Req {climate_class}' for climate_class in unique_climate_classes] + ['All Climates']
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels)

        #ax.set_xticklabels(['Req 0', 'Req 1', 'Req 2', 'Req 3', 'Req 4'])
        ax.set_ylabel(f'{result_type}')
        plt.xticks(rotation=45)

        # Add legend
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ['BayesDPS', 'NormDPS'])

    plt.tight_layout()
    if save_plot:
        file_name = f"{plot_type}plot_comparison_under_testsets.png"
        full_path = os.path.join(folder_path, file_name)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.show()









def plot_multiple_results_not_used(bayes_result, norm_result, label, result_types, climate_classes, save_plot=True, box=True):
    """
    Create a box plot comparing multiple key results for Bayes and Norm policies across different climate sets.

    Parameters:
    - bayes_result: Dictionary of results for the Bayes policy across scenarios.
    - norm_result: Dictionary of results for the Norm policy across scenarios.
    - label: Label for the output directory.
    - result_types: List of result types to plot (e.g., ['installed_capacity', 'Jplan', 'Jdeficit']).
    - require_0: List of indices characterizing climates that require 0 capacity.
    - require_1: List of indices characterizing climates that require 1 capacity.
    - require_2: List of indices characterizing climates that require 2 capacities.
    - save_plot: Whether to save the plot.
    - box: Whether to plot box or violin plot.
    """

    folder_path = f'results/figures/{label}'
    os.makedirs(folder_path, exist_ok=True)

    plt.rcParams.update({'font.size': 16})  # Adjust the base font size

    

    fig, axs = plt.subplots(len(result_types), 1, figsize=(12, 8 * len(result_types)))

    for idx, result_type in enumerate(result_types):
        ax = axs[idx]
        
        # Group results as dictionary

        # Extract results for each climate set
        data = []
        labels = []
        positions = []
        colors = []

        for climate_set_idx, climate_set in enumerate(climate_sets):
            bayes_data = [bayes_result[result_type][i] for i in climate_set]
            norm_data = [norm_result[result_type][i] for i in climate_set]

            data.append(bayes_data)
            labels.append(f'Bayes {climate_labels[climate_set_idx]}')
            colors.append('blue')
            data.append(norm_data)
            labels.append(f'Norm {climate_labels[climate_set_idx]}')
            colors.append('pink')
        
        for i in range(len(climate_sets)):
            positions.extend([i * 2, i * 2 + 0.5])
        
        if box:
            # Create box plot
            for i, (box_data, color) in enumerate(zip(data, colors * len(climate_labels))):
                bp = ax.boxplot(box_data, positions=[positions[i]], widths=0.4, patch_artist=True)
                for element in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
                    plt.setp(bp[element], color=color)
                plt.setp(bp['boxes'], facecolor=color, alpha=0.5)
                
                # Add points to represent the costs
                for point in box_data:
                    ax.scatter(positions[i], point, color=color, alpha=0.7)
        
            # Customize the plot
            ax.set_title(f'Comparison of {result_type} for Bayes and Norm Policies Across Different Climate Sets')
            ax.set_xlabel('Climate Sets')
            ax.set_xticks([0.25, 2.25, 4.25])
            ax.set_xticklabels(climate_labels)
            ax.set_ylabel(f'{result_type}')
            plt.xticks(rotation=45)

            # Add legend
            handles = [plt.Line2D([0], [0], color='blue', lw=4),
                    plt.Line2D([0], [0], color='pink', lw=4)]
            ax.legend(handles, ['BayesDPS', 'NormDPS'])

        else:
            # Create violin plot
         
            for idx, result_type in enumerate(result_types):
                ax = axs[idx] if len(result_types) > 1 else axs
                # Prepare data for plotting
                data = []
                classes = []
                alive = []

                for climate_set_idx, climate_set in enumerate(climate_sets):
                    for i in climate_set:
                        data.append(bayes_result[result_type][i])
                        classes.append(climate_labels[climate_set_idx])
                        alive.append('Bayes')

                        data.append(norm_result[result_type][i])
                        classes.append(climate_labels[climate_set_idx])
                        alive.append('Norm')

                plot_data = pd.DataFrame({
                    'data': data,
                    'class': classes,
                    'alive': alive
                })

                # Create violin plot
                sns.violinplot(data=plot_data, x="class", y="data", hue="alive", split=True, bw = 0.1, inner="quart", ax=ax)
                sns.swarmplot(data=plot_data, x="class", y="data", hue="alive", dodge=True, color='k', alpha=0.5, ax=ax)

                # Customize the plot
                ax.set_title(f'Comparison of {result_type} for Bayes and Norm Policies Across Different Climate Sets')
                ax.set_xlabel('Climate Sets')
                ax.set_ylabel(result_type)

                # Increase font size
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                # Add legend
                handles = [plt.Line2D([0], [0], color='blue', lw=4),
                        plt.Line2D([0], [0], color='pink', lw=4)]
                ax.legend(handles, ['BayesDPS', 'NormDPS'])


    plt.tight_layout()

    file_name = f"boxplot_comparison_across_climates.png"
    full_path = os.path.join(folder_path, file_name)
    if save_plot:
        plt.savefig(full_path, dpi=100)
    plt.show()





    


def plot_hv(bayes_hv, norm_hv, label):
    folder_path = f'results/figures/{label}'
    os.makedirs(folder_path, exist_ok=True)

    # Determine consistent y-axis range
    #min_hv = min(min(bayes_hv), min(norm_hv))
    #max_hv = max(max(bayes_hv), max(norm_hv))+ 0.1
    min_hv = 0.2
    max_hv = 0.95

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Plot Bayes HV
    axs[0].plot(bayes_hv, marker='o', linestyle='-')
    axs[0].set_title('Bayes Hypervolume over Generations')
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Hypervolume (HV)')
    axs[0].grid(True)
    axs[0].set_ylim(min_hv, max_hv)

    # Plot Norm HV
    axs[1].plot(norm_hv, marker='o', linestyle='-')
    axs[1].set_title('Norm Hypervolume over Generations')
    axs[1].set_xlabel('Generation')
    axs[1].grid(True)
    axs[1].set_ylim(min_hv, max_hv)

    plt.tight_layout()
    plt.savefig(f'{folder_path}/{label}_hypervolume_convergence.png')
    plt.show()


def make_pareto_table(bayesDPS_objs,normDPS_objs, config_bayesDPS, config_normDPS):

    bayesDPS_sorted = sorted(bayesDPS_objs, key=lambda x: x[1])
    normDPS_sorted = sorted(normDPS_objs, key=lambda x: x[1])

    # Creating a table with the sorted data
    df_BayesDPS = pd.DataFrame(bayesDPS_sorted, columns=[f'{config_bayesDPS}_Jplanning', f'{config_bayesDPS}_Jdefict'])
    df_NormDPS = pd.DataFrame(normDPS_sorted, columns=[f'{config_normDPS}_Jplanning', f'{config_normDPS}_Jdefict'])

    # Merging the dataframes for a comparative table
    pareto_table = pd.concat([df_NormDPS, df_BayesDPS], axis=1)

    return pareto_table


def plot_objectives(objectives_over_time):
    plt.figure(figsize=(10, 6))
    num_generations = len(objectives_over_time)
    generations = range(num_generations)

    # Assuming objectives_over_time is a list of lists, 
    # where each inner list contains the objective values of all solutions in one generation
    for i in range(len(objectives_over_time[0][0])):  # assuming all generations have the same number of objectives
        objective_values = [generation[i] for generation in objectives_over_time for generation in generation]
        print(objective_values)
        plt.plot(generations, objective_values, marker='.', linestyle='--', label=f'Objective {i+1}')


    plt.title('Objectives Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/figures/objectives_convergence.png')
    plt.show()

    
    #x=range(1,H+1)
    #r_d = mat.repmat(rd,1, H)
    #r = [ r_d.tolist()[0], rc, rgi, rt, rgw,  rswp, r_mw ]
    #plt.stackplot(x, r, labels=['Desal', 'Cachuma', 'Gibraltar', 'Tunnel', 'Groundwater', 'State Water Project', 'Market Water'])
    #plt.legend(loc = 'lower left')
    #plt.xlabel('Time')
   # plt.ylabel('Demand [AFd]')
    #plt.title('Water Demand')
    # plt.savefig('../figures/demand_carry.png',  dpi=100)
    #plt.show()
         
    
    #sw_c, sw_g, sw_t, gw_c, dw_c, swp_c, mw_c, distr_c = sim.cost_traj(rc, rgi, rgw, sgw, rd, rswp, r_mw, rt)
    #c_d = mat.repmat(dw_c, 1, H)
    # dis_c = mat.repmat(distr_c, 1, H)
    #plt.stackplot(x, c, labels=['Desal',  'Cachuma', 'Gibraltar', 'Tunnel', 'Groundwater', 'State Water Project', 'Market Water'])
    #plt.legend(loc = 'upper left')
    #plt.title('Cost')
    #plt.xlabel('Time')
    #plt.ylabel('Cost [$]')

    # plt.savefig('../figures/cost_carry.png',  dpi=100)
    #plt.show()
    
   
    
    #with open('../traj/mw_c' + s + '.txt', 'w') as filehandle:
        #for listitem in mw_c:
            #filehandle.write('%s\n' % listitem)


