import numpy as np
import pandas as pd
import argparse
import policy
#import utils_GPR as ut

### extract indicator function
### map rbf policy output to capacity  
### will be used in plot_optimization.py




def extract_climate_indicators(climate_means, climate_stds, t, Nyears_per_cycle, normalize_min_max):
    """
    Extracts climate indicators based on given climate means, standard deviations, and time step.
    
    # input. climate_means is i slice of loaded_mu, 2d matrix for ith simulation :Nyears*Ncycles
    # input. t: simulation time step in months.
    # output: climate indicators: mean(t), mean(t2100), std(t2100)
    """
    T = int(t / 12)
    cycle_index = int(T / Nyears_per_cycle)
    T_convert = T + 125
    year2100_idx = np.where(np.arange(1875, 2275) == 2100)[0][0]

    mean_t = climate_mean_avg(climate_means, 0, T_convert, cycle_index)
    mean_2100 = climate_mean_avg(climate_means, 0, year2100_idx, cycle_index)
    std_2100 = climate_stds[year2100_idx, cycle_index]

    norm_mean_t = normalize_climate_indicators(mean_t, normalize_min_max['mean'])
    norm_mean_2100 = normalize_climate_indicators(mean_2100, normalize_min_max['mean'])
    norm_std_2100 = normalize_climate_indicators(std_2100, normalize_min_max['std'])

    return [norm_mean_t, norm_mean_2100, norm_std_2100]


def climate_mean_avg(climate_means, t_plus, T_convert, cycle_index):
    """
    Calculates the average of climate means around a specified index.
    
    :param climate_means: Array of mean climate data
    :param t_plus: Adjustment to the index T_convert
    :param T_convert: Converted index based on time t
    :param cycle_index: Current cycle index based on update frequency
    :return: Average climate mean
    """
    T = T_convert + t_plus
    start_idx = max(0, T - 4)
    means = climate_means[start_idx:T + 1, cycle_index]
    return np.mean(means)


def normalize_climate_indicators(indicator, min_max):
    """
    Normalizes an indicator value between a specified range.
    
    :param indicator: The indicator value to normalize
    :param min_max: Tuple of (min, max) values for normalization
    :return: Normalized indicator
    """
    min_value, max_value = min_max
    normalized_indicator = (indicator - min_value) / (max_value - min_value)
    return normalized_indicator








  


def classify_trends(trends):
    classifications = {'Increasing': [], 'Decreasing': [], 'Inc-Dec': [], 'Dec-Inc': []}
    
    for idx, trend in enumerate(trends):
        # Smooth the trend for better classification
        smoothed = savgol_filter(trend, 21, 3)  # Window size 21, polynomial order 3  # # Choosing an odd number, about 20% of your data length
        first_half_trend = np.diff(smoothed[:len(smoothed)//2])
        second_half_trend = np.diff(smoothed[len(smoothed)//2:])

        if np.all(first_half_trend > 0) and np.all(second_half_trend < 0):
            classifications['Inc-Dec'].append(idx)
        elif np.all(first_half_trend < 0) and np.all(second_half_trend > 0):
            classifications['Dec-Inc'].append(idx)
        elif np.all(np.diff(smoothed) > 0):
            classifications['Increasing'].append(idx)
        elif np.all(np.diff(smoothed) < 0):
            classifications['Decreasing'].append(idx)

    return classifications

def select_representatives(classifications, trends, n_for_each):   # select the time series that has the most prominant trends characterstics
    representatives = {}
    for key, indices in classifications.items():
        representatives[key] = []
         # Check if there are enough series to select from
        if len(indices) >= n_for_each:
            # Calculate the sum of absolute changes for each series in the category
            changes = [(idx, np.sum(np.abs(np.diff(trends[idx])))) for idx in indices]
            # Sort the list by changes in descending order to get the strongest trends first
            sorted_changes = sorted(changes, key=lambda x: x[1], reverse=True)
            # Select the top n_for_each indices
            top_indices = [trends[idx] for idx, change in sorted_changes[:n_for_each]]
            representatives[key].extend(top_indices)
        else:
            # If not enough series, take all available
            representatives[key].extend([trends[idx] for idx in indices])

 
    return representatives
