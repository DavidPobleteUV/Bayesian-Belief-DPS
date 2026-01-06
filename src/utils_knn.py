# Copyright (c) 2024 Mofan Zhang

import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import seaborn as sns


# month_Knn: disaggregate from annual to monthly using monthly obs proportions

#input: annQsyn, one synthetic time series of 100year, annual mean with trend  2000- 2009 year
#input: monthly_obs, monthly observation data, 1860-1950
#input:syn_index: which synthetic time series?
#input: frac for detrend function

#output: monthly time series array, 150year * 12, both with trend and detrend

# np.random.seed(66)

def month_knn(annQsyn,monthly_obs, syn_index, frac):

  #step1: operate on observation time series

  # compute annual mean of monthly obs
  annQ = compute_annual_mean(monthly_obs)
  #monthly proportion matrix
  P_monthly = monthly_p(monthly_obs)

  #step2: operate on annQsyn

  #detrend
  de_annQsyn = detrend(annQsyn, frac)['detrend']
  #rescale based on annQ mean and std
  de_annQsyn= (de_annQsyn - np.mean(de_annQsyn))/ np.std(de_annQsyn) * np.std(annQ)+ np.mean(annQ)

  # step3: resample using KNN

  N = len(annQ)
  K = int(np.sqrt(N))
  NS = len(de_annQsyn)

    # init
  de_monQsyn = pd.DataFrame(np.zeros((NS, 12)))
  monQsyn = pd.DataFrame(np.zeros((NS, 12)))

  for i in range(NS):

    # for each value in ann_syn, find k nearest neighbors in ann_obs, according to distance between de_annQsyn[i] and annQ
    # k nearest neighbors in the historical record
    knn_index = np.argsort(np.abs(de_annQsyn[i] - annQ))[0:K]
    # weighted resample an index from knn_index
    sample_index =  weighted_sample(K, knn_index)
    # fill monPsyn[i,] based on monthly proportion matrix at year = sample_index
    de_monQsyn.iloc[i, :]  =  de_annQsyn[i] * P_monthly.iloc[sample_index, :] * 12
    monQsyn.iloc[i, :] = annQsyn[i] *  P_monthly.iloc[sample_index, :] * 12
  
  # step4: convert to array
  monQsyn = np.array(monQsyn.values.flatten() )
  de_monQsyn =  np.array(de_monQsyn.values.flatten() )

  
  return monQsyn , de_monQsyn


##helper

def detrend(x, frac):
    """
    Detrend a time series using LOESS.
    
    Parameters:
    - x: The time series data (list or numpy array).
    - frac: The fraction of the data used when estimating each y-value in the LOWESS.  #locally weighted scatterplot smoothing
    
    Returns:
    - A DataFrame with the trend line and detrended data.
    """
  
    # Generate an index array
    index = np.arange(len(x))
    
    # Apply LOWESS smoothing
    # The 'frac' parameter controls the degree of smoothing
    lowess_result = lowess(x, index, frac=frac)
    
    # Extract smoothed values
    x_smoothed = lowess_result[:, 1]
    
    # Calculate the detrended series
    x_detrended = x - x_smoothed
    
    # Create a DataFrame to store the trend line and detrended data
    detrend_df = pd.DataFrame({'trend_line': x_smoothed, 'detrend': x_detrended})
    
    return detrend_df

#KNN helper
def weighted_sample(k, ordered_index):
    # Build kernel
    w = 1 / np.arange(1, k + 1)

    w = w / np.sum(w)  # Normalize to create a probability vector

    # Cumulative sum to build CDF
    w_cumsum = np.cumsum(w)

    # Generate a uniform random number in [0,1]
    uni = np.random.uniform(0, 1)

    # Map that random number to which index
    # Find where the uniform random number would fit in the cumulative distribution
    map_index = np.searchsorted(w_cumsum, uni)

    # Adjust if the map_index is equal to k, which is out of the index range
    map_index = min(map_index, k - 1)

    # Get the actual sample index from the ordered index
    sample_index = ordered_index[map_index]

    return sample_index

#compute monthly proportion matrix

def monthly_p(monthly_obs):
    
    monthPmat = np.array(monthly_obs).reshape(-1, 12) 
    # Compute the sum for each 12 month (row sum)
    annSum = monthPmat.sum(axis=1).reshape(-1, 1)  # Reshape for broadcasting
    # Handle potential division by zero if any FiveSum is 0
    annSum[annSum == 0] = np.nan
   
    #compute proportion matrix
    P_month = monthPmat/ annSum
    tol = .0001
    assert np.sum(np.abs(np.sum(P_month, 1) - 1) > tol) == 0, 'rows should sum to 1'

    P_month_df = pd.DataFrame(P_month)

    return P_month_df

# helper: given a monthly time series, compute its 12-month chunk mean == annual mean
def compute_annual_mean(obs):

    ann_mean = []
    num_periods = len(obs)/12

    for i in range(int(num_periods)):
        start_idx = i*12
        end_idx = start_idx + 12
        chunk = obs[start_idx : end_idx]
        chunk_mean = chunk.mean()
        ann_mean.append(chunk_mean)
    
    ann_mean = np.array(ann_mean)  # convert to array

    return ann_mean
