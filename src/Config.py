
# -*- coding: utf-8 -*-
"""
@author: mofanzhang
Config.py saves the configuration for the simulation
# test9 climate

"""
import numpy as np
import pandas as pd


class Config:
    def __init__(self, run_bayes = True, option_type = 'both'):


        self.result_loc = 'data/Annual_GPR'
        self.updated_freg = 1 # update every # years

        climate_label = 'test9'
        self.syn_data_mat = f'{climate_label}_runoff.csv'
        self.indicator_mat = f'{climate_label}_inflow_indicators.csv'
        self.bayes_mat = f'{climate_label}_bayes_indicators.npz'
        self.prep_mean = f'{climate_label}_prep_mean.csv'
        self.prep_std = f'{climate_label}_prep_std.csv'


        if run_bayes :
            self.BayesDPS = True
        else:
            self.BayesDPS = False
        
        # Option type can be 'both', 'static', or 'flexible'
        self.option_type = option_type

        self.Ny = 100
        self.H = 1200


        self.normalize_min_max = {
        'mean': (-3, 3),
        'std': (0.2, 1.2)
        }


        self.sizes = [0.5, 1, 1.5, 2, 3]  #MCM    capacity options, small, medium, large
        self.fixed_cost = [2000, 1800, 1600, 1400, 1200] #per MCM of capacity
        self.flex_cost = [2450, 1980, 1740, 1540, 1340] #per MCM of capacity
        self.flex_expand = 1200 #per MCM of expanding capacity
        self.static_cost = [2000, 2000, 2000, 2000, 2000]  

        #self.sizes = [0.5, 1, 1.5, 2]  #MCM    capacity options, small, medium, large
        #self.fixed_cost = [2000, 1800, 1600, 1400] #per MCM of capacity
        #self.fixed_cost = [2000,2000, 2000, 2000]
        #self.flex_cost = [2450, 1980, 1740, 1540] #per MCM of capacity
        #self.flex_expand = 1200 #per MCM of expanding capacity


        self.demand = 5.5 # MCM/month
     

        self.operate = 0.5   # /MCM/month

        try:
            self.inflows = pd.read_csv(f'{self.result_loc}/{self.syn_data_mat}', header = None).T/12
        except FileNotFoundError:
            print(f"File not found: {self.result_loc}/{self.syn_data_mat}")
        
        self.inflow_indicators = pd.read_csv(f'{self.result_loc}/{self.indicator_mat}')

        self.climate_means =  np.load(f'{self.result_loc}/{self.bayes_mat}')['mu']  # bayes indicators
        self.climate_sigmas = np.load(f'{self.result_loc}/{self.bayes_mat}')['std']
        self.prep_means = pd.read_csv(f'{self.result_loc}/{self.prep_mean}',index_col = 0)    # non-bayesian prep indicators
        self.prep_sigmas = pd.read_csv(f'{self.result_loc}/{self.prep_std}',index_col = 0)

        self.initial_storage  = 45
        self.hedge_param = 15
        self.smax = 60    #   # existing reservoir max storage
