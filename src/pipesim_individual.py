#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mofanzhang
pipesim_individual.py simulates the a water system given a pipe expansion
decision policy and a single hydrological scneario and returns trajectories of 
relevant hydrological variables and costs

created on Feb 24 Sat 2024
last updated on Mar 21 Thur 2024
"""

import numpy as np
import numpy.matlib as mat
import numba
from numba import njit
import random
from policy import *



class log_results:
    pass
    class traj:
        pass
    class cost:
        pass


@njit
def nsim_equiv_res(s, u, n, s_max):
    # extremely fast reservoir mass balance computation
    r         = max(0, min(u, s))
    s_        = s + n - r
    s_        = max(0, min(s_, s_max ))

    return s_, r

class ReservoirSimulation:
    def __init__(self, hedge_param, smax):
        self.hedge_param = hedge_param
        self.smax = smax

    def operate(self, s_next, demand_, inflow):   #inflow at time t
        hedge = self.hedge_param * s_next
        flood = 10 * s_next * (1 / 2 - demand_ / self.smax) - 9 * (self.smax / 2 - demand_) + demand_
        u = max(min(hedge, demand_), flood)

        # Reservoir mass balance
        s_next, release = nsim_equiv_res(s_next, u, inflow, self.smax,)
        water_def = max(0, demand_ - release)
        if water_def < 1e-4:
            water_def = 0

        return s_next, release, water_def
    

class Pipe_sim(object):
   ############# define relevant class parameters
    def __init__(self, opt_par, config):
        
        self.Ny          = config.Ny #number of years
        self.H           = config.H # length of time horizon
        self.nsim = opt_par.nsim   # simualte under nsim hydroligical scenarios

        self.BayesDPS = config.BayesDPS
        #self.with_flex = config.flex
        self.option_type = config.option_type

        self.fixed_cost = config.fixed_cost #per MCM of capacity
        self.flex_cost = config.flex_cost  #per MCM of capacity
        self.flex_expand = config.flex_expand  # per MCM of expanding capacity
        self.static_cost = config.static_cost
        self.operate = config.operate   # /MCM/month
        self.sizes = config.sizes     
        self.flex_built = False
        self.fix_built = False

        self.normalize_min_max = config.normalize_min_max
        self.year2100_idx = np.where(np.arange(1875, 2275) == 2099)[0][0]
    

    
        self.demand      = config.demand
        self.inflows = config.inflows   # 100, 100-year monthly inflow time series
        self.indicators = config.inflow_indicators    # rolling mean over previous monthly inflow 
        self.climate_means = config.climate_means   # 100 realizations of climate indicators updated over time
        self.climate_sigmas = config.climate_sigmas
        self.prep_means = config.prep_means
        self.prep_sigmas = config.prep_sigmas

        self.updated_freg = config.updated_freg   


        self.N           = opt_par.N #hidden nodes
        self.M           = opt_par.M #inputs
        self.K           = opt_par.K #outputs
         # init the neural network
        self.nn = None # neural network object

        self.initial_storage = config.initial_storage  # Assuming S0 is defined in config
        self.hedge_param = config.hedge_param  # Assuming hedge_param is defined in config
        self.smax = config.smax  # existing reservoir max storage
    

    def simulation(self, P,s):   #s: index of climate scenarios

        # Set up the neural network from parameters
        self.nn = set_param(P, self.M, self.N, self.K) 

        ######## prepare output fields
        log                  = log_results()
        log.capacity = []
        log.storage = []
        log.release = []
        log.deficit_annual = []
        log.deficit = []
        log.Jdef = []
        log.Jplan  = []
        log.inflow = []
        log.flex_built = False
        log.fix_built = False
        log.timing = []  # List to record times when capacity is built or expanded
      
        #add flex_built and fix_built (bool variable) in result log



        ######## initialize vectors for simulation
        H = self.H
        demand = self.demand
        inflow = self.inflows.iloc[:,s].values
        indicator = self.indicators.iloc[:,s].values
        climate_mean = self.climate_means[s,...]
        climate_sigma = self.climate_sigmas[s,...] 
        prep_mean = self.prep_means.iloc[:,s].values
        prep_sigma = self.prep_sigmas.iloc[:,s].values
        #mean_ts = self.bayes_indicators['mean_ts'][:,s]   # annual indicators
        #mean_t50s = self.bayes_indicators['mean_t50s'][:,s]
        #mean_t2100 = self.bayes_indicators['mean_t2100'][:,s]
        #std_t2100 = self.bayes_indicators['std_t2100'][:,s]

        smax = self.smax  # existing reservoir max storage

        self.flex_built = False
        self.fix_built = False
        installed_capacity = np.zeros(H)
        plan_cost = np.zeros(H)
        deficit = np.zeros(H)
        Jdeficit = np.zeros(H)
        s_next = np.zeros(H+1)  #storage
        s_next[0] = self.initial_storage
        r = np.zeros(H)
   

        for t in range(H) : # simulate 100-year monthly horizon

            demand_ = demand - installed_capacity[t]

            ################## system already has a reservoir , simulate reservoir mass balance (simple hedge rule)
            # reservoir mass balance (simple hedging rule)  # or just release the water available (ref: 266G)
            res_simulation = ReservoirSimulation(self.hedge_param, self.smax)
            s_, release, water_def = res_simulation.operate(s_next[t], demand_, inflow[t])
            s_next[t+1] = s_
            r[t] = release
            deficit[t]= water_def
            Jdeficit[t] = deficit[t] ** 2
             ##########################################
            #evaluation policy and action update (extract actions from policy and simulate effect)
            
            ###### policy indicators at time t
            if t > 60 and t % 12 == 0 : # after 5 years, start to evaluate the policy
                
                
                #storage_t = s_next[t+1]    # current time step storage (#or, using average of last 12-month storage)
                inflow_t = indicator[t]  # current time step inflow indicator (rolling mean over last 5 years), normalized
        
                if self.BayesDPS:
                    climate_indicators = self.extract_climate_indicators(climate_mean, climate_sigma, t)
                    indicators = [inflow_t] + climate_indicators  #need to norm inputs
                else:
              
                    #indicators = [inflow_t] 
                    prep_indicators = [prep_mean[t//12], prep_sigma[t//12]]
                    indicators = [inflow_t] + prep_indicators
                    
                ##### extract action from policy and simualte effect
                #capacity = get_output(indicators, self.nn)[0]  # only one output from nnet
                #planning_cost, installed_capa = self.Policy2Action(capacity, installed_capacity[t])

                # if using softmax
                capacity = get_output(indicators, self.nn) #here it's a probability distribution  
                planning_cost, installed_capa = self.Policy2Action2(capacity, installed_capacity[t])
        
            
                plan_cost[t] = planning_cost
                installed_capacity [t:H] = installed_capa

                # Record the time when capacity is built or expanded
                if installed_capa != 0 and (len(log.timing) == 0 or installed_capa != installed_capacity[t-1]):
                    log.timing.append(t)

    
        Jdef = np.sum(Jdeficit[60:])
        Jplan = np.sum(plan_cost) + self.operate * np.sum(installed_capacity)
        deficit_annual = np.sum(deficit.reshape(-1,12), axis = 1) 

        ######## write vectors to output
        log.inflow = inflow
        log.capacity = installed_capacity
        log.storage = s_next[1:]
        log.release = r
        log.deficit_annual = deficit_annual
        log.deficit = deficit
        log.Jdef = Jdef
        log.Jplan  = Jplan
        log.flex_built = self.flex_built
        log.fix_built = self.fix_built
    
        return log


    def map_capacity_to_built(self, capacity,lower_thres ,upper_thres, seg):
        # Determine capacity_to_built based on the capacity range using a conditional expression
        # the number of options is determined by len(self.sizes)

        n_options = len(self.sizes)  # number of capacity options
        # Determine capacity_to_built based on the capacity range using a conditional expression
        if lower_thres < capacity <= upper_thres:
            for i in range(n_options):
                if lower_thres + seg * i < capacity <= lower_thres + seg * (i + 1):
                    return self.sizes[i]
        else:
            return None
        
    

    # include flexible options
    def Policy2Action(self, capacity, capa_installed):
        
        # depend on the option type, develop different actions [both; static only; flexible only]
        #### evaluate output from rbf (0-1):
        n_options = len(self.sizes)
        seg = 1 / (n_options * 2 + 1) if self.option_type == 'both' else 1 / (n_options + 1)
    
        planning_cost = 0

        if capacity <= seg:  # do nothing
            pass

        elif self.option_type in ['both', 'static'] and seg < capacity <= seg * (n_options + 1):  # Install fixed pipe

            capa_to_built = self.map_capacity_to_built(capacity, seg ,seg * (n_options + 1), seg)

            if not self.fix_built and not self.flex_built : 

                self.fix_built = True
                capa_installed = capa_to_built

                fixed_cost = None
                for i, size in enumerate(self.sizes):
                    if size == capa_to_built:
                        fixed_cost = self.fixed_cost[i]
                        break  
                planning_cost += fixed_cost * capa_installed

        elif  self.option_type == 'both' and seg * (n_options + 1) < capacity <= 1:  # Install flexible pipe (both options available)

            capa_to_built = self.map_capacity_to_built(capacity, seg * (n_options + 1) , 1, seg)

            if not self.fix_built:  

                if not self.flex_built : #build pipe the very first time
                    
                    self.flex_built = True
                    capa_installed = capa_to_built

                    flex_cost = None
                    for i, size in enumerate(self.sizes):
                        if size == capa_to_built:
                            flex_cost = self.flex_cost[i]
                            break 
                    planning_cost += flex_cost * capa_installed

                
                elif capa_to_built > capa_installed:
                        planning_cost = self.flex_expand *(capa_to_built -capa_installed )
                        capa_installed = capa_to_built

        elif self.option_type == 'flexible' and seg < capacity <= seg * (n_options + 1):  # Install flexible pipe (only flexible option)
            
            capa_to_built = self.map_capacity_to_built(capacity, seg, seg * (n_options + 1), seg)
            if not self.fix_built:
                if not self.flex_built:  # Build pipe the very first time
                    self.flex_built = True
                    capa_installed = capa_to_built
                    flex_cost = None
                    for i, size in enumerate(self.sizes):
                        if size == capa_to_built:
                            flex_cost = self.flex_cost[i]
                            break 
                    planning_cost += flex_cost * capa_installed
                
                elif capa_to_built > capa_installed:
                    planning_cost = self.flex_expand * (capa_to_built - capa_installed)
                    capa_installed = capa_to_built

        return planning_cost, capa_installed
    

    def Policy2Action2(self, probabilities, capa_installed):
        """
        Convert the policy probabilities to actions.

        Parameters:
        - probabilities: The policy probabilities.
        - capa_installed: The installed capacity.

        Returns:
        - action: The action to take.
        """
        action_idx = np.argmax(probabilities)
        planning_cost = 0
        n_sizes = len(self.sizes)

        #if action_idx == 0:
        # Assuming index 0 corresponds to the "do nothing" action
            #pass

        if self.option_type == 'flexible':
            # Handle flexible-specific logic

            capa_to_built = self.sizes[action_idx]
            flex_cost = self.flex_cost[action_idx]
            if not self.fix_built:

                if not self.flex_built: # First-time build
                    self.flex_built = True
                    capa_installed = capa_to_built
                    planning_cost += flex_cost * capa_installed
                elif capa_to_built > capa_installed:  # Capacity expansion
                    planning_cost += self.flex_expand * (capa_to_built - capa_installed)
                    capa_installed = capa_to_built

        elif self.option_type == 'both':
            # Separate handling for static and flexible within 'both'
            if action_idx < n_sizes:
                # Static actions
                capa_to_built = self.sizes[action_idx]
                if not self.fix_built and not self.flex_built:
                    self.fix_built = True
                    capa_installed = capa_to_built
                    planning_cost += self.fixed_cost[action_idx] * capa_installed
            else:
                # Flexible actions
                flex_idx = action_idx - n_sizes
                capa_to_built = self.sizes[flex_idx]
                if not self.fix_built:
                    if not self.flex_built:
                        self.flex_built = True
                        capa_installed = capa_to_built
                        planning_cost += self.flex_cost[flex_idx] * capa_installed
                    elif capa_to_built > capa_installed:
                        planning_cost += self.flex_expand * (capa_to_built - capa_installed)
                        capa_installed = capa_to_built

        elif self.option_type == 'static':

             if action_idx < n_sizes:
                static_idx = action_idx
                capa_to_built = self.sizes[static_idx]
                if not self.flex_built:

                    if not self.fix_built:    # first-time built
                        self.fix_built = True
                        capa_installed = capa_to_built
                        planning_cost += self.static_cost[static_idx] * capa_installed
                    elif capa_to_built > capa_installed: # Capacity expansion
                        planning_cost += self.static_cost[static_idx] * (capa_to_built - capa_installed)
                        capa_installed = capa_to_built
            
        return planning_cost, capa_installed


    
    def extract_climate_indicators(self, climate_means, climate_stds, t):

        # input. climate_means is i slice of loaded_mu, 2d matrix for ith simulation :Nyears*Ncycles
        # input. t: simulation time step in months.
        # output: climate indicators: mean(t), mean(t2100), std(t2100)

        # get the year index
        T = int(t/12)
        #which update cycle
        cycle_index = int(T/self.updated_freg)     

        T_convert = T + 125
        mean_t = self.climate_mean_avg(climate_means, 0 ,T_convert, cycle_index)
        #mean_t25 = self.climate_mean_avg(climate_means, 25, T_convert, cycle_index)
        mean_2100 = self.climate_mean_avg(climate_means, 0 , self.year2100_idx, cycle_index)
        std_2100 = climate_stds[self.year2100_idx, cycle_index]

        #normalized
        norm_mean_t = self.normalize_climate_indicators(mean_t,  'mean')
        #norm_mean_t25 = self.normalize_climate_indicators(mean_t25, 'mean')
        norm_mean_2100 = self.normalize_climate_indicators(mean_2100,  'mean')
        norm_std_2100 = self.normalize_climate_indicators(std_2100, 'std')

    

        return [norm_mean_2100, norm_std_2100]   
    

    def climate_mean_avg(self, climate_means, t_plus,T_convert, cycle_index ):

        T = T_convert + t_plus
        start_idx = max(0, T - 4)
        means = climate_means[start_idx:T + 1, cycle_index]
        return np.mean(means)

    
    def normalize_climate_indicators(self, indicator,  indicator_type):

        min_value, max_value = self.normalize_min_max[indicator_type]
        normalized_indicator = (indicator - min_value) / (max_value - min_value)

        return normalized_indicator

    

