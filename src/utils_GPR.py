# Copyright (c) 2023 Mofan Zhang
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.linalg import cholesky, cho_solve,cho_factor,solve_triangular


#########
#GPR function
#obs: a vector for virtual observations, whole time periods 
#GCM_data: a matrix. ncols = number of GCMs to develop priors
#Output: posterior mean, variance, log marginal likelihood

def GPR_func(obs,Nobs, GCM_data, sigma1, sigma2, sigma3, sigma_n, use_cholesky = False):

    # prepending length of observed data to GCM_Data
    tmp = np.vstack((GCM_data.iloc[:Nobs,:], GCM_data))
    smoothedMean = np.mean(tmp, axis = 1 )

    #develop kernel
    KGCM = compute_KGCM(GCM_data, Nobs)
    KNoise = compute_KNoise(GCM_data, Nobs, sigma1, sigma2, sigma3)
    K = KGCM + KNoise
    #Kernel must be positive semidefinite

    #prepare data for GPR
    y = obs[:Nobs]
    mu = smoothedMean[:Nobs] #prior estimate mean(up to observations)
    mu_s = smoothedMean[Nobs:]  # prior estimate mean (whole study period)
    Noise = sigma_n * np.eye(Nobs, Nobs)  # noise only add on diagonal 
 
    #subset kernel
    K_o = K[:Nobs, :Nobs]
    K_s = K[Nobs:, Nobs:]
    K_os = K[:Nobs, Nobs:]

    #compute the difference between prior mean and obs
    diff = y - mu      # observations deviates from its prior estimate mean
    if diff.ndim == 1:  
        #diff = diff[:, np.newaxis]   # convert diff into a 2D array, make sure dimension match in later matrix multiplication
        diff = np.array(diff).reshape(-1, 1)  # Convert to numpy array and reshape.
    ###############################################################
    # compute posteior mean and variance, log marginal likelihood

    if use_cholesky == False:

        eigenvalues = np.linalg.eigvalsh(K_o + Noise)
        assert np.all(eigenvalues >= 0)
        f = mu_s + K_os.T @ np.linalg.inv(K_o + Noise) @ diff.ravel()
        Sigma_s = K_s - K_os.T @ np.linalg.inv(K_o + Noise) @ K_os
        lml = -0.5 * (diff.T @ np.linalg.inv(K_o + Noise) @ diff + np.log(np.linalg.det(K_o + Noise))).item() # convert to a scaler /omit the constant term since it does not impact hyperparameter selection

    if use_cholesky == True:
        A = K_o + Noise
        eigenvalues = np.linalg.eigvalsh(A)  #make sure A is positive semi-definite
        #print("Smallest eigenvalues:", np.sort(eigenvalues)[:10])
        assert np.all(eigenvalues >= 0)

        alpha, v, L = Choleskey_trick(A, diff ,K_os)
        # Compute the posterior mean f
        f = mu_s + np.dot(K_os.T, alpha).ravel()    
        # Compute the posterior covariance Sigma_s
        Sigma_s = K_s - np.dot(v.T, v)
        #compute log marginal likelihood
        lml = -0.5 * (np.dot(diff.T, alpha) + 2*np.sum(np.log(np.diag(L)))).item()  #convert to a scaler 


    # prepare output
    prior_mu = mu_s
    prior_cov = K_s
    post_mu = f
    post_cov = Sigma_s


    return prior_mu, prior_cov, post_mu, post_cov,lml


#####################################################
#Tune GPR hyperparameters function
def tune_GPR(sig1, sig2,sig3, mod_ids, Nobs,FiveYrDf, smoothed_50yrDf,FiveYr_anom ,use_cholesky = False, Fiveyr = False):

    Marginal_Likelihood = np.zeros((len(sig1), len(sig2), len(sig3)))
    
    for mod_id in mod_ids:   #mod_ids = [1:16]

        oos_gcm = FiveYrDf.iloc[:, mod_id].values
        column_indices = [i for i in range(smoothed_50yrDf.shape[1]) if i != 0 and i != mod_id]  # all columns except the first(year column) and the oos-gcm
        if Fiveyr == True: 
            GCM_data = smoothed_50yrDf.iloc[::5, column_indices]  # extract these columns and every 5th row
        else:
            GCM_data = smoothed_50yrDf.iloc[:, column_indices]
        sigma_n = FiveYr_anom.iloc[:Nobs, mod_id].var()

        # prepedning observed data to GCM_data
        tmp = np.vstack((GCM_data.iloc[:Nobs,:], GCM_data))

        # compute smoothed covariance and mean
        smoothedCov = np.cov(tmp, rowvar = True)
        smoothedMean = np.mean(tmp, axis = 1 )

        length = len(smoothedMean)
        col_index, row_index = np.meshgrid(np.arange(length),np.arange(length))  # will be used for computing noise matrix for kernel

        for ii in range(len(sig1)):
            for jj in range(len(sig2)):
                for kk in range(len(sig3)):

                    sigma1 = sig1[ii]
                    sigma2 = sig2[jj]
                    sigma3 = sig3[kk]

                    Noise_matrix = sigma1**2 * np.exp(-(1/(2*sigma2**2)) *(col_index - row_index) **2) + sigma3**2 * np.eye(length)
                   
                     # compute kernel
                    Kernel = (16/15)*smoothedCov + Noise_matrix 

                    # observations
                    y = oos_gcm[:Nobs]
                    Noise = sigma_n * np.eye(Nobs, Nobs)  ###noise only add on diagonal

                    K_o = Kernel[:Nobs, :Nobs]

                    # compute marginal likelihood
                
                    mu = smoothedMean[:Nobs] # prior estimate mean (up to observations)
                    diff = y - mu      # observations deviates from its prior estimate mean
                    if diff.ndim == 1:  
                        diff = diff[:, np.newaxis]   # convert diff into a 2D array, make sure dimension match in later matrix multiplication
                        
                    if use_cholesky == True:
                        A = K_o + Noise
                        c,low = cho_factor(A, lower = True)
                        alpha = cho_solve((c, low), diff)
                        L = cholesky(A, lower=True) # this is the lower triangular cholesky matrix
                        #compute log marginal likelihood
                        tmp = -0.5 * (np.dot(diff.T, alpha) + 2*np.sum(np.log(np.diag(L)))).item()  #convert to a scaler 


                    if use_cholesky == False:
                        tmp = -0.5 * (diff.T @ np.linalg.inv(K_o + Noise) @ diff + np.log(np.linalg.det(K_o + Noise))).item() 

                    if np.isreal(tmp):
                        Marginal_Likelihood[ii, jj, kk] += tmp
                    else:
                        Marginal_Likelihood[ii, jj, kk] = -np.inf
                

    max_idx = np.argmax(Marginal_Likelihood)  # the flat index of the maximum value
    #print(Marginal_Likelihood.shape)
    a,b,c = np.unravel_index(max_idx, Marginal_Likelihood.shape)
    #print(a,b,c)
    Max_Likelihood_sigma1 = sig1[a]
    Max_Likelihood_sigma2 = sig2[b]
    Max_Likelihood_sigma3 = sig3[c]

    return Max_Likelihood_sigma1, Max_Likelihood_sigma2, Max_Likelihood_sigma3

def compute_KGCM(GCM_data, Nobs):

    # prepending observed data to GCM_Data
    tmp = np.vstack((GCM_data.iloc[:Nobs,:], GCM_data))

    # compute smoothed covariance and mean
    smoothedCov = np.cov(tmp, rowvar = True)
    
    KGCM = (16/15)*smoothedCov

    return KGCM

def compute_KNoise(GCM_data, Nobs, sigma1, sigma2, sigma3):   # this is a RBF(short-term autocorrelations) + white noise 
     
     # create noise matrix
    length = len(GCM_data) + Nobs
    col_index, row_index = np.meshgrid(np.arange(length),np.arange(length))
    Noise_matrix = sigma1**2 * np.exp(-(1/(2*sigma2**2)) *(col_index - row_index) **2) + sigma3**2 * np.eye(length)


    return Noise_matrix

def Choleskey_trick(A, b, K_os):
     # solve Ax = b for x, x = A-1b
     # solve Av = K_os for v
     
     c,low = cho_factor(A, lower = True)
     x = cho_solve((c, low), b)
     L = cholesky(A, lower=True) # this is the lower triangular cholesky matrix
     v = solve_triangular(L, K_os, lower= True)
    
     return x, v,L

###########################################################
#synthetic generator algorithm
# Function: ensemble_generator()
#generate non-stationary time series
#generate noisy non-stariony time series with fixed observation noise

def ensemble_generator(N_ts,GCM_data, obs, Nobs, Nyears_per_cycle, Ncycles, sigma1, sigma2, sigma3, sigma_n,obs_smoothed_50yr, with_noise):

    # Init result matrix

    synthetic_data_mat = np.zeros((Ncycles * Nyears_per_cycle, N_ts))   #2D
    Nyears= len(obs)  # length of the whole study period (1875-2275)
    #Init null matrix with zeros for records of post_mu, post_std in each update, for each synthetic realization
    mu_mat= np.zeros((N_ts, Nyears, Ncycles ))    #3D
    std_mat = np.zeros((N_ts, Nyears, Ncycles ))

    # 0. GPR first update, use obs (up to year Nobs)
    prior_mu, prior_cov, post_mu, post_cov, lml = GPR_func(obs, Nobs, GCM_data, sigma1, sigma2, sigma3, sigma_n,use_cholesky = True)

    # sample 100 sequences based on updated model 0 for year (for example) 1980-2000
    sliced_mu = post_mu[Nobs : Nobs + Nyears_per_cycle]   # slice post_mu according to the intended generate period (1975-2000, if nobs starts from 1975)
    sliced_cov = post_cov[Nobs : Nobs+Nyears_per_cycle, Nobs : Nobs + Nyears_per_cycle]
    # sigma_n is observation noise
    sliced_cov_with_noise = sliced_cov.copy() + sigma_n * np.eye(len(sliced_cov), len(sliced_cov)) 
    # Now generate  synthetic time series with the added noise in the covariance matrix
    obs_mat = np.random.multivariate_normal(sliced_mu, sliced_cov_with_noise, N_ts)
    
    # each row of syn_mat is the virtual observations

    #1. perform update for each virtual observations, repeat for Ncycles
    
    for ts in range(N_ts):

        updated_obs = obs[:Nobs]  
        updated_Nobs = Nobs

        for i in range(Ncycles):
            if i ==0:
                vir_obs = obs_mat[ts, :]
            else:
                vir_obs = synthetic_data
            
            updated_obs = np.concatenate([updated_obs, vir_obs])  # update observations with virtual observations
            updated_Nobs = updated_Nobs + Nyears_per_cycle
            #updated_sigma_n = (updated_obs - obs_smoothed_50yr[:updated_Nobs]).var()
           

            # GPR update
            prior_mu, prior_cov, post_mu, post_cov, lml = GPR_func(updated_obs, updated_Nobs, GCM_data, sigma1, sigma2, sigma3, sigma_n,use_cholesky = True)

            # slice mu, cov, according to the prediction period(=periods to generate sysnthetic data)
            sliced_mu = post_mu[updated_Nobs: updated_Nobs + Nyears_per_cycle]   # slice post_mu according to the intended generate period (2000-2020, if updated_Nobs starts from 2000)
            sliced_cov = post_cov[updated_Nobs: updated_Nobs + Nyears_per_cycle, updated_Nobs : updated_Nobs + Nyears_per_cycle]
            sliced_cov_with_noise = sliced_cov.copy() +sigma_n * np.eye(len(sliced_cov), len(sliced_cov))

            # sample 1 realization of sequence
            synthetic_data =  np.random.multivariate_normal(sliced_mu , sliced_cov_with_noise).flatten()  # GPR assumes data distribute by multivaraite_gaussian. the number of points to be generated are based on the length of sliced_mu, sliced_cov
            
            # attach generated data to result mat; record post_mu, post_cov to ith column to result matrix


            synthetic_data_mat[i * Nyears_per_cycle : (i + 1) * Nyears_per_cycle, ts] = synthetic_data
            mu_mat[ts, :, i] = post_mu
            if with_noise:

                std_mat[ts,:, i] = np.sqrt(np.diag(post_cov) + sigma_n )
            else:
                std_mat[ts,:, i] = np.sqrt(np.diag(post_cov))
    
    return synthetic_data_mat,mu_mat,std_mat



#Function: ensemble_generator_with_n()
#generate noisy non-stationary time series with changing observation noise , also noisy cliamte indicators

def ensemble_generator_with_n(N_ts,GCM_data, obs, Nobs, Nyears_per_cycle, Ncycles, sigma1, sigma2, sigma3, sigma_n,obs_smoothed_50yr, with_noise):

    # Init result matrix

    synthetic_data_mat = np.zeros((Ncycles * Nyears_per_cycle, N_ts))   #2D
    Nyears= len(obs)  # length of the whole study period (1875-2275)
    #Init null matrix with zeros for records of post_mu, post_std in each update, for each synthetic realization
    mu_mat= np.zeros((N_ts, Nyears, Ncycles ))    #3D
    std_mat = np.zeros((N_ts, Nyears, Ncycles ))

    # 0. GPR first update, use obs (up to 1980)
    prior_mu, prior_cov, post_mu, post_cov, lml = GPR_func(obs, Nobs, GCM_data, sigma1, sigma2, sigma3, sigma_n,use_cholesky = True)

    # sample 100 sequences based on updated model 0 for year 1980-2000
    sliced_mu = post_mu[Nobs : Nobs + Nyears_per_cycle]   # slice post_mu according to the intended generate period (1975-2000, if nobs starts from 1975)
    sliced_cov = post_cov[Nobs : Nobs+Nyears_per_cycle, Nobs : Nobs + Nyears_per_cycle]
    # sigma_n is observation noise
    sliced_cov_with_noise = sliced_cov.copy() + sigma_n * np.eye(len(sliced_cov), len(sliced_cov)) 
    # Now generate  synthetic time series with the added noise in the covariance matrix
    obs_mat = np.random.multivariate_normal(sliced_mu, sliced_cov_with_noise, N_ts)
    
    # each row of syn_mat is the virtual observations

    #1. perform update for each virtual observations, repeat for Ncycles
    
    for ts in range(N_ts):

        updated_obs = obs[:Nobs]  
        updated_Nobs = Nobs

        for i in range(Ncycles):
            if i ==0:
                vir_obs = obs_mat[ts, :]
            else:
                vir_obs = synthetic_data
            
            updated_obs = np.concatenate([updated_obs, vir_obs])  # update observations with virtual observations
            updated_Nobs = updated_Nobs + Nyears_per_cycle
            updated_sigma_n = (updated_obs - obs_smoothed_50yr[:updated_Nobs]).var()
           

            # GPR update
            prior_mu, prior_cov, post_mu, post_cov, lml = GPR_func(updated_obs, updated_Nobs, GCM_data, sigma1, sigma2, sigma3, updated_sigma_n,use_cholesky = True)

            # slice mu, cov, according to the prediction period(=periods to generate sysnthetic data)
            sliced_mu = post_mu[updated_Nobs: updated_Nobs + Nyears_per_cycle]   # slice post_mu according to the intended generate period (2000-2020, if updated_Nobs starts from 2000)
            sliced_cov = post_cov[updated_Nobs: updated_Nobs + Nyears_per_cycle, updated_Nobs : updated_Nobs + Nyears_per_cycle]
            sliced_cov_with_noise = sliced_cov.copy() + updated_sigma_n * np.eye(len(sliced_cov), len(sliced_cov))

            # sample 1 realization of sequence
            synthetic_data =  np.random.multivariate_normal(sliced_mu , sliced_cov_with_noise).flatten()  # GPR assumes data distribute by multivaraite_gaussian. the number of points to be generated are based on the length of sliced_mu, sliced_cov
            
            # attach generated data to result mat; record post_mu, post_cov to ith column to result matrix


            synthetic_data_mat[i * Nyears_per_cycle : (i + 1) * Nyears_per_cycle, ts] = synthetic_data
            mu_mat[ts, :, i] = post_mu
            if with_noise:

                std_mat[ts,:, i] = np.sqrt(np.diag(post_cov) + updated_sigma_n )
            else:
                std_mat[ts,:, i] = np.sqrt(np.diag(post_cov))
    
    return synthetic_data_mat,mu_mat,std_mat