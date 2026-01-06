
# -*- coding: utf-8 -*-

"""
@author: mofanzhang
simulation.py simulates the a water system given a dam expansion
decision policy and a set of hydrological scenarios and returns objective values

**Last update: 4/5/2024**  
This notebook is for developing V2 water supply planning model for Bayesian DPS  

**V1 cast study**  
Planning horizon: 2000- 2100, 100-yr, monthly 
Infrastructure options:   
a.	A static dam . Once built, cannot be expanded later 
b.	A flexible dam . Could be expanded later
Only allow choose from a or b. only one site for building infrastrucutre 

Action: at each time t, decide:  
a.	Do nothing  
b.	Build new static dam (decide capacity from small, medium and large)  
c.	Build new flexible dam
d.	Expand existing flexible dam

"""
import numpy as np
import numpy.matlib as mat
import numba
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

class Dam(object):
   ############# define relevant class parameters
    def __init__(self, opt_par, config):
        
        self.Ny          = config.Ny #number of years
        self.H           = config.H # length of time horizon
        self.nsim = opt_par.nsim   # simualte under nsim hydroligical scenarios

        self.fixed_cost = config.fixed_cost #per MCM of capacity
        self.flex_cost = config.flex_cost  #per MCM of capacity
        self.flex_expand = config.flex_expand  # per MCM of expanding capacity
        self.sizes = config.sizes     #e.g.[10, 30, 50]
        self.flex_built = False
        self.fix_built = False
    
        self.demand      = config.demand
        self.inflows = config.inflows   # 100, 100-year monthly inflow time series
        self.climate_means = config.climate_means   # 100 realizations of climate indicators updated over time
        self.climate_sigmas = config.climate_sigmas
        self.updated_freg = config.updated_freg    # 20year


        self.N           = opt_par.N #hidden nodes
        self.M           = opt_par.M #inputs
        self.K           = opt_par.K #outputs

        self.initial_storage = config.initial_storage  # Assuming S0 is defined in config
        self.hedge_param = config.hedge_param  # Assuming hedge_param is defined in config
        self.smax = config.smax  # existing reservoir max storage


    def simulation(self, P):

        #extract and interpret RBF paramters from policy param list P
        param, lin_param = set_param(P, self.N, self.M, self.K)
        planning_costs = []   # list of values for each simulation episode under 100 climate scenarios.
        def_costs = []
        sizes = self.sizes

        smax = self.smax  # existing reservoir max storage


        #Simulate 
        for i in range(self.nsim):  # simulate under nsim cliamte scenarios   

            ##### init parameters for simulation
            inflow = self.inflows.iloc[:,i].values
            climate_mean = self.climate_means[i,...]
            climate_sigma = self.climate_sigmas[i,...] 
            s_next = self.initial_storage

            planning_cost = 0
            def_cost = 0
            self.flex_built = False
            self.fix_built = False
            installed_capa = 0 # dam capacity installed in the system
        

            for t in range(self.H) : # simulate 100-year monthly horizon

                demand_ = self.demand 
            
                ## if system already built a reservoir , simulate reservoir mass balance (simple hedge rule. opimized for the size)
                if installed_capa > 0: 
                    dead_storage = 20
                    samx = installed_capa - 20
                    # reservoir mass balance (simple hedging rule)  # or just release the water available (ref: 266G)
                    # suppose sizes = [50, 80, 100]
                    # optimzied hedge_param = [10,20,30]. # here, find the hedge_param corresponding to the size 
                    hedge_param = self.hedge_param[sizes.index(installed_capa)]
                    hedge = self.hedge_param * s_next
                    flood = 10 * s_next *(1/2 - demand_ /smax) - 9*(smax/2 - demand_) + demand_
                    u = max(min(hedge,demand_), flood)

                    #reservoir mass balance
                    s_next, release = nsim_equiv_res(s_next, u, inflow[t], smax)
                    water_def = max(0, demand_ - release)
                    if water_def < 1e-4:
                        water_def = 0
               
                ##################################   
                # 2. Compute current water deficit
                if t > 60 : #spin up time (5 year)
                    def_cost += water_def ** 2

                ##########################################
                # 3. evaluation policy and action update (extract actions from policy and simulate effect)
                
                ###### policy indicators at time t
                storage_t = s_next    # current time step storage (#or, using average of last 12-month storage)
                climate_indicators = self.extract_climate_indicators(climate_mean, climate_sigma, t)
                #need norm inputs
                size_max = max(sizes)
                indicators = [storage_t/smax, installed_capa/size_max] + climate_indicators 
                #obtain output from rbf function
                capacity = get_output(indicators, param, lin_param, self.N, self.M, self.K)[0]  # only one output from rbf
                planning_cost, installed_capa = self.Policy2Action(capacity, planning_cost, installed_capa)    # capacity to build (whether fix or flex); update installed_capa; update planning cost

            
            # Apeend costs for the current simulation scenario

            def_costs.append(def_cost)
            planning_costs.append(planning_cost)


        # across nsim cliamte scenarios, compute average objective
        Jplan= np.mean(planning_costs)
        Jdef = np.mean(def_costs)
        #print('pipe_simulation:Jdef', Jdef)

        return Jplan, Jdef    # return planning cost and water deficit cost


    def map_capacity_to_built(self, capacity,lower_thres ,upper_thres, seg):
        
        [s_size, m_size, l_size] = self.sizes

        # Ensure the capacity falls within the expected range
        if not (lower_thres < capacity <= upper_thres):
            raise ValueError(f"Capacity {capacity} is out of the expected range.")

    # Determine capacity_to_built based on the capacity range using a conditional expression
            # Determine capacity_to_built based on the capacity range using a conditional expression
        if lower_thres < capacity <= lower_thres + seg:
            return s_size
        elif lower_thres + seg < capacity <= lower_thres + seg * 2:
            return m_size
        elif lower_thres + seg * 2 < capacity <= upper_thres:
            return l_size
        else:
            return None



    def Policy2Action(self, capacity, planning_cost, capa_installed):
            
        #### evaluate output from rbf (0-1):
        seg = 1/7
        if capacity <= seg:  # do nothing
            pass

        elif seg < capacity <= seg*4 :    # install fix pipe

            capa_to_built = self.map_capacity_to_built(capacity, seg ,seg*4, seg)

            if not self.fix_built and not self.flex_built : 

                self.fix_built = True
                capa_installed = capa_to_built
                planning_cost += self.fixed_cost * capa_installed

        elif  seg*4 < capacity <= seg*7 :  #install flex pipe

            capa_to_built = self.map_capacity_to_built(capacity, seg*4 ,seg*7, seg)

            if not self.fix_built:  

                if not self.flex_built : #build pipe the very first time
                    
                    self.flex_built = True
                    capa_installed = capa_to_built
                    planning_cost += self.flex_cost * capa_installed
                
                elif capa_to_built > capa_installed:
                        planning_cost += self.flex_expand *(capa_to_built -capa_installed )
                        capa_installed = capa_to_built

        return planning_cost, capa_installed





    def extract_climate_indicators(self, cliamte_means, cliamte_stds, t):

        # input. cliamte_means is i slice of loaded_mu, 2d matrix for ith simulation
        # input. t: simulation time step in months.
        # output: climate indicators: mean(t), mean(t+50), std(t+50)

        # get the year index
        T = int(t/12)
        #which update cycle
        cycle_index = int(T/self.updated_freg)    # updated_freg = 20

        #convert year index in cliamte_means file (calimte_means file year start from 1875)
        T_convert = T + 125
        
        mean_t = self.climate_mean_avg(cliamte_means, 0 ,T_convert, cycle_index)
        mean_t_50 = self.climate_mean_avg(cliamte_means, 50, T_convert, cycle_index)
        std_t_50 = cliamte_stds[T_convert + 50, cycle_index]

        norm_mean_t = self.normalize_climate_indicators(mean_t, mean = True)
        norm_mean_t_50 = self.normalize_climate_indicators(mean_t_50, mean = True)
        norm_std_t_50 = self.normalize_climate_indicators(std_t_50, mean = False)

        return [norm_mean_t, norm_mean_t_50, norm_std_t_50]   

    def climate_mean_avg(self, climate_means, t_plus,T_convert, cycle_index ):
        means = [climate_means[T_convert + t_plus, cycle_index]]
        if T_convert - 4 >=0:
            means.append(climate_means[T_convert - 1, cycle_index])
            means.append(climate_means[T_convert - 2, cycle_index])
            means.append(climate_means[T_convert - 3, cycle_index])
            means.append(climate_means[T_convert - 4, cycle_index])
        mean_avg = np.mean(means)

        return mean_avg
    
    def normalize_climate_indicators(self, indicator, mean = True ):

        if mean == True:

            min_value = - 0.3
            max_value = 1.1
        
        else:
            min_value = 0.98
            max_value = 1.2
        normalized_indicator = (indicator - min_value) / (max_value - min_value)
        return normalized_indicator