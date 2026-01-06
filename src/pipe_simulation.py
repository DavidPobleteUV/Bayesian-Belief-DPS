
# -*- coding: utf-8 -*-

"""
@author: mofanzhang
simulation.py simulates the a water system given a pipe expansion
decision policy and a set of hydrological scenarios and returns objective values

**Last update: 1/29/2025**  
This notebook is for developing V1 water supply planning model for Bayesian DPS  

**V0 cast study**  
Planning horizon: 2000- 2100, 100-yr, monthly 
There is an existing storage in the system 
Infrastructure options:   
a.	A fixed pipe . Once built, cannot be expanded later 
b.	A flexible pipe . Could be expanded later
Only allow choose from a or b. only one site for building infrastrucutre 

Action: at each time t, decide:  
a.	Do nothing  
b.	Build new static pipe (decide capacity from small, medium and large)  
c.	Build new flexible pipe
d.	Expand existing flexible pipe 

"""
import numpy as np
import numpy.matlib as mat
import numba
import pandas as pd
from numba import njit
import random
from policy import *




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

    


class Pipe(object):
   ############# define relevant class parameters
    def __init__(self, opt_par, config):
        
        self.Ny          = config.Ny #number of years
        self.H           = config.H # length of time horizon
        self.nsim = opt_par.nsim   # simualte under nsim hydroligical scenarios
        self.BayesDPS = config.BayesDPS   # if True, run BayesDPS; else run normDPS
        #self.with_flex = config.flex   # if True, include flexible pipe option
        self.option_type = config.option_type  # if 'both', include both fixed and flexible options; else, only fixed or flex option

        self.fixed_cost = config.fixed_cost #per MCM of capacity
        self.flex_cost = config.flex_cost  #per MCM of capacity
        self.flex_expand = config.flex_expand  # per MCM of expanding capacity
        self.static_cost = config.static_cost
        self.operate = config.operate   # /MCM/month
        self.sizes = config.sizes     #e.g.[10,20,30]
        self.flex_built = False
        self.fix_built = False

        
        self.normalize_min_max = config.normalize_min_max
        self.year2100_idx = np.where(np.arange(1875, 2275) == 2099)[0][0]
    
    
        self.demand      = config.demand
        self.updated_freg = config.updated_freg 

        self.N           = opt_par.N #hidden nodes
        self.M           = opt_par.M #inputs
        self.K           = opt_par.K #outputs
        # init the neural network
        self.nn = None   

        self.initial_storage = config.initial_storage  # Assuming S0 is defined in config
        self.hedge_param = config.hedge_param  # Assuming hedge_param is defined in config
        self.smax = config.smax  # existing reservoir max storage

        self.inflows = config.inflows   # 100, 100-year monthly inflow time series
        self.indicators = config.inflow_indicators    # rolling mean over previous monthly inflow 
        self.climate_means = config.climate_means   # N realizations of climate indicators updated over time
        self.climate_sigmas = config.climate_sigmas
        self.prep_means = config.prep_means   # non-bayesian prep indicators， annual
        self.prep_sigmas = config.prep_sigmas
    
    

    def simulation(self, P):

        #extract and interpret Nnet paramters from policy param list P
        # Set up the neural network from parameters
        self.nn = set_param(P, self.M, self.N, self.K)  

        planning_costs = []   # list of values for each simulation episode under 100 climate scenarios.
        def_costs = []

        smax = self.smax  # existing reservoir max storage
        demand = self.demand
        H = self.H


        #Simulate 
        for i in self.nsim:  # simulate under nsim cliamte scenarios 

            #print(f"Simulating scenario {i + 1}/{self.nsim}")  

            ##### init parameters for simulation
            inflow = self.inflows.iloc[:,i].values
           
            indicator = self.indicators.iloc[:,i].values
            climate_mean = self.climate_means[i,...]
            climate_sigma = self.climate_sigmas[i,...] 
            prep_mean = self.prep_means.iloc[:,i].values
            prep_sigma = self.prep_sigmas.iloc[:,i].values
            
         
            planning_cost = 0
            def_cost = 0
            self.flex_built = False
            self.fix_built = False
          

            installed_capacity = np.zeros(H)
            plan_cost = np.zeros(H)
            deficit = np.zeros(H)
            Jdeficit = np.zeros(H)
            s_next = np.zeros(H+1)  #storage
            s_next[0] = self.initial_storage
            r = np.zeros(H)
            printed = False
            

            for t in range(self.H) : # simulate 100-year monthly horizon

                demand_ = demand - installed_capacity[t]
            
                ############## system already has a reservoir , simulate reservoir mass balance (simple hedge rule)
                
                # reservoir mass balance (simple hedging rule)  # or just release the water available (ref: 266G)
                res_simulation = ReservoirSimulation(self.hedge_param, self.smax)
                s_, release, water_def = res_simulation.operate(s_next[t], demand_, inflow[t])
                s_next[t+1] = s_
                r[t] = release
                deficit[t]= water_def
                Jdeficit[t] = deficit[t] ** 2
    

                ##########################################
                # 3. evaluation policy and action update (extract actions from policy and simulate effect)
                
                ###### policy indicators at time t
                if t>60 and t % 12 == 0: #spin up time (5 year)
                    #evaluate policy only at the beginning of each year

                    #storage_t = s_next[t+1]    # current time step storage (#or, using average of last 12-month storage)
                    inflow_t = indicator[t]  # current time step inflow indicator (rolling mean over last 5 years), normalized
                

                    if self.BayesDPS:
                        climate_indicators = self.extract_climate_indicators(climate_mean, climate_sigma, t)
                        indicators = [inflow_t] + climate_indicators
                      
                    else:
                        prep_indicators = [prep_mean[t//12], prep_sigma[t//12]]
                        indicators = [inflow_t] + prep_indicators
                       
                        
                    #obtain output from nnet function

                    #capacity = get_output(indicators, self.nn)[0]  # only one output from nnet
                    #planning_cost, installed_capa = self.Policy2Action(capacity, installed_capacity[t])

                    # if using softmax
                    capacity = get_output(indicators, self.nn) #here it's a probability distribution  
                    planning_cost, installed_capa = self.Policy2Action2(capacity, installed_capacity[t])

                    #if self.BayesDPS:
                        #plan_cost[t] = planning_cost * (1 + 0.5 * uncertainty)  # add uncertainty to planning cost)
                    #else:
                    
                    plan_cost[t] = planning_cost
               
                    installed_capacity [t:H] = installed_capa

                
            # Append costs for the current simulation scenario
            def_cost = np.sum(Jdeficit[60:])
            planning_cost = np.sum(plan_cost) + self.operate * np.sum(installed_capacity)
            def_costs.append(def_cost)
            planning_costs.append(planning_cost)
            #print('climate scenario:', i, 'planning cost:', planning_cost, 'deficit cost:', def_cost)
        


        # across nsim cliamte scenarios, compute average objective
        Jplan= np.mean(planning_costs)
        Jdef = np.mean(def_costs)
        #print('pipe_simulation:Jdef', Jdef)
        #print('pipe_simulation:Jplan', Jplan)

        return Jplan, Jdef    # return planning cost and water deficit cost


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
            # Handle static-only logic: sequential build, same unit cost for development and expansion
            if action_idx <= n_sizes:
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
    






    

    
        