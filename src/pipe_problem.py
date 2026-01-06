
# -*- coding: utf-8 -*-
"""
@author: mofanzhang
pipe_problem.py defines the evaluate function that calls the simulation function for every individual and collects the objective values.  
created on Feb 25 Sun 16:52:40 2024
last update on 

"""
from __future__ import absolute_import, division, print_function

import math
import numpy as np
import random
import operator
import functools
from platypus.core import Problem, Solution, EPSILON
from platypus.types import Real, Binary
from abc import ABCMeta
import pipe_simulation

class Pipe_Problem(Problem):
     
    def __init__(self, opt_param, config, nconstrs = 0):

        # Initialize the problem with the number of decision variables and objectives
        super(Pipe_Problem, self).__init__(nvars = opt_param.nparam, nobjs = opt_param.nobjs, nconstrs = 0, function=None)
        # Define the lower and upper bounds for each decision variable
       
        self.types[:] = [Real(lb, ub) for lb, ub in zip(opt_param.LB, opt_param.UB)]
        self.model = pipe_simulation.Pipe(opt_param, config)
         # Set the direction of optimization for each objective
        # MINIMIZE (default) or MAXIMIZE can be set for each objective
        #self.directions[:] = [Problem.MINIMIZE, Problem.MINIMIZE]  # Assuming both J1 and J2 should be minimized
  
        
    def evaluate(self, individual): #individual is a string of param, sim contains all methods and objects
        
        #extract decision variables from the solution
        str_param = individual.variables
        #apply policy in the simulator and return objectives
        J1, J2 = self.model.simulation(str_param)
        # update solution objectives
        individual.objectives = [J1, J2]
        

    def random(self):
        solution = Solution(self)
        solution.variables[:self.nobjs-1] = [random.uniform(0.0, 1.0) for _ in range(self.nobjs-1)]
        solution.variables[self.nobjs-1:] = 0.5
        solution.evaluate()
        return solution



