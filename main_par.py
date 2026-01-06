# -*- coding: utf-8 -*-
"""
Created on Wed July 31 10:00:00 2024
last update on Mon Aug 26 15:22:00 2024
This script is used to run the optimization across nseeds in parellel
for 01 experiment: where type = 'both'
for normDPS and bayesDPS both have 3 indicators
@author: mofanzhang
"""

import sys
import os
sys.path.append('src')
import numpy as np
import numpy.matlib as mat
import matplotlib.pyplot as plt
import pandas as pd

from multiprocessing import Pool
import functools
import time


import pipe_simulation
from pipe_simulation import Pipe
from pipesim_individual import Pipe_sim
from pipe_problem import Pipe_Problem
from plot_optimization import *
from Config import *
from src import *

import argparse
from platypus import *
from platypus.core import nondominated
from platypus import Hypervolume
from platypus.experimenter import experiment, calculate, display
from concurrent.futures import ProcessPoolExecutor
import pickle
from platypus import ProcessPoolEvaluator
import configparser
import logging
import csv


class OptimizationParameters(object):
    def __init__(self, run_bayes = True):
        ###DEMO OPTIMIZATION, use higher values of max_gen and npop if results are not converged

        #self.nsim = np.array([465, 276, 203, 418, 137, 378, 487, 77, 353, 201, 217, 541, 401, 568, 467, 404, 117, 573, 164, 0, 69, 579, 562, 199, 144, 4, 1, 92, 186, 101]) #training 30 climates in test_4
        #self.nsim = np.array([548, 376, 380, 565, 230, 354, 428, 547, 207, 245, 443, 554, 109, 570, 504, 467, 73, 50, 123, 23, 194, 3, 531, 562, 436, 83, 125, 130, 98, 27]) #training 30 climates in test_4
        #self.nsim = np.array([441, 311, 470, 409, 404, 303, 390, 318, 339, 471, 494, 383, 289,248, 245, 243, 252, 210, 162, 199, 137, 194,193,174,3,98,97,54,85,2]) #training 30 climates in test_5
        #self.nsim = np.array([490, 242, 200, 192, 159, 443, 415, 247, 147, 208, 109, 187, 193,5, 129,  14,  64, 497, 349, 107, 285, 141, 256,  71,  59, 169,269,  76, 378, 417, 105, 221,  98,  11, 114, 209, 380, 194, 350,21,  67, 217,   2, 137,  27, 118,  45, 108,  53, 439]) #50 cliamtes in training in test_6
        #self.nsim = np.array([ 86,  94, 153, 190, 130,  29, 102,  18, 115,  45,  43,  22,  66, 141,  44, 129,   9,  89, 179,  65,  90,  19, 258, 168,  48, 271,14, 221,  88, 163, 299, 173, 174, 278, 294, 234, 296, 230,  55,245, 286, 284, 241, 248, 280, 157, 220, 275, 272, 213])    # 50 climate in test8
        self.nsim = np.array([ 81,  85,  93,  94,   6,  67,  16,  78, 175,  41, 118,  49,   3,119,   1, 152,  15,  52,  86,  13, 155,  76, 199, 231, 112, 259,156, 186,  79, 252, 139, 286, 182, 176, 270, 283, 272, 285, 192,287, 275, 160, 180, 298, 167, 214, 281, 249, 263, 290])   # 50 climate in test9
    #     self.nsim = np.array([283, 186, 131, 181, 102, 118, 175, 203, 272, 193,  24, 337,  55,
    #    223, 158, 242, 155, 132, 174, 275, 106, 451, 484, 318, 256, 289,
    #    436, 347,  67, 240,  78, 235, 359, 425, 456, 474, 314,   2, 217,
    #    377, 428, 260, 437, 243, 405,  51, 468, 483, 494, 282])  #50 in test7
        self.max_gen  = 450 # GA number of generations; 300
        self.npop     = 180 #GA population size ; 360 or 80
        self.nfe      =  81000  #81000  #self.max_gen*self.npop   check for 3000, 4000,difference?
        self.use_parallel = True 
        self.cores    = 4
        self.nseeds   = 8  #use 15

        # Adjust aprameters based on whether running BayesDPS or not
        if run_bayes:
            self.N = 14 # hidden notes for BayesDPS   N = M+k+1; % number of basis is set following rule of thumb: #inputs + #outputs + 1
            self.M = 3 # inputs for BayesDPS
        else:
            self.N = 14 # hidden notes for normDPS
            self.M = 3 # inputs for normDPS
       
        self.K        = 10 #outputs
        self.nparam   = self.N * (self.M + 1 + self.K) + self.K
        self.nobjs    = 2

        self.LB = -1 * np.ones(self.nparam)
        self.UB = 1 * np.ones(self.nparam)

        self.log_freq = 500  #2000


class Solution():
    pass



def run_single_experiment(seed, opt_par, config):
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    with ProcessPoolEvaluator(opt_par.cores) as evaluator:

        random.seed(seed)
        start_time = time.time()
        #logging.info(f"Starting experiment for seed {seed} at {start_time}")
        algorithm = EpsMOEA(Pipe_Problem(opt_par, config), epsilons = [10], population_size=opt_par.npop, offspring_size=opt_par.npop, evaluator=evaluator, log_frequency=opt_par.log_freq, verbose=3)
        algorithm.run(opt_par.nfe)
        end_time = time.time()

        #logging.info(f"Finished experiment for seed {seed} at {end_time}, duration: {end_time - start_time}")
        return algorithm.result
    

def run(opt_par, config):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    print(f"Running {opt_par.nseeds} experiments in parallel, each using {opt_par.cores} cores")
    partial_run = functools.partial(run_single_experiment, opt_par=opt_par, config=config)

    with ProcessPoolExecutor(max_workers=opt_par.nseeds) as executor:
        futures = [executor.submit(partial_run, seed) for seed in range(opt_par.nseeds)]
        seed_results = [future.result() for future in futures]
    
    # Structure the results in the same format as the original code
    results = {'EpsMOEA': {'Pipe_Problem': {i: solutions for i, solutions in enumerate(seed_results)}}}
    
    return results

 

if __name__ == '__main__':

    #set configuration
    
    option_type = 'both'  #  'both', 'static', 'flexible'
    label = '111401'


    #plotting config
    plot_compare = True
    config_bayesDPS = f"bayes_{label}"
    config_normDPS = f"norm_{label}"


    ###############

   
    for run_bayes in [True, False]:

        opt_par = OptimizationParameters(run_bayes)
        config = Config(run_bayes, option_type)
        

        if run_bayes:
            config_name = config_bayesDPS
        else:
            config_name = config_normDPS
        print(f'running optimization for {config_name}')
    
        # run optimization and save result 
        result = run(opt_par, config)

        objs  = []
        seed_objs = []
        param = []

        #reference_point = [1e4, 1e4]
        reference_point = [6000, 3000]
        hv = Hypervolume(minimum=[0, 0], maximum=reference_point) 
        hvs = []

        for seed in range(opt_par.nseeds):
            solutions = result['EpsMOEA']['Pipe_Problem'][seed]
            objs.extend([ [s.objectives[0], s.objectives[1]] for s in solutions]) 
            seed_obj = [[s.objectives[0], s.objectives[1]] for s in solutions]
            seed_objs.append(seed_obj)
            param.extend([s.variables for s in solutions])  # check
            front = [s for s in solutions] # non-dominated solutions found so far during the optimziation ptocess
            if front:
                hvs.append(hv.calculate(front))
        #print("main hypervolumnes for different seeds:", hvs)
        # Save hypervolume for the best across all seeds
            best_hv = max(hvs) if hvs else None

        mask = is_pareto_efficient(np.array(objs))

        objs_eff   = []
        param_eff  = []
        for m, o, p in zip(mask, objs, param):
            if m:
                objs_eff.append(o)
                param_eff.append(p)

        #find no deficit solution
        idx_nodeficit = np.argmin(np.array(objs_eff)[:,1] )

        #creat solution structure
        solution = Solution()
        solution.best_score = objs_eff[idx_nodeficit]
        solution.best_solution = param_eff[idx_nodeficit]
        solution.all_solutions = param_eff
        solution.objs = objs_eff
        solution.seed_objs = seed_objs 
        solution.best_hv = best_hv  # Save the best hypervolume value for this run
        #solution.log = []
        #print("main solution.best_socore:",solution.best_score)
    
        
        # dictionary with list object in values
        details = {
            'fixed_cost' : [config.fixed_cost],
            'flex_cost' : [config.flex_cost],
            'flex_expand': [config.flex_expand],
            'sizes'    : [config.sizes],

            'reservoir max storage'         : [config.smax],
            'reservoir hedge param'        : [config.hedge_param],
        
            'demand' : [config.demand],
            'nfe' : [opt_par.nfe],
            'max_gen': [opt_par.max_gen],
            'npop': [opt_par.npop]
        }
        
        # creating a Dataframe object
        df = pd.DataFrame(details)
        solution.config = df



        #if run on supercomputer:
        output_dir = "results"  # realtive path from the project directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        string = os.path.join(output_dir, f'results_{config_name}.dat')
    
        with open(string, 'wb') as f:
            pickle.dump(solution, f)
        
        #print('main objs_eff:',objs_eff)