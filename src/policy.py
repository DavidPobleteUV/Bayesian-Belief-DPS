# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 20:00:00 2024

@author: mofan zhang

policy.py implements the methematical formulation of the nerual network which takes as input the policy paramters and produces as output the expansion decisions

reates a neural network class NeuralNetwork and includes functions to initialize it and perform a forward pass. The set_param function adjusts the neural network parameters 
from a flattened array that would be suitable for optimization with evolutionary algorithms.
"""

import numpy as np
import logging

def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    z = np.exp(x)
    return z / (z + 1)

def sigmoid(x):
    #to prevent overflow
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    # empty contains juke hence will be faster to allocate than zeros
    result = np.empty_like(x)
    result[pos_mask] = _positive_sigmoid(x[pos_mask])
    result[neg_mask] = _negative_sigmoid(x[neg_mask])
    return result


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def relu(x):
    return np.maximum(0, x)

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
    
    def forward(self, x):
        # Forward pass through the network
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        max_z = np.max(self.z2, axis=1, keepdims=True)  # Get max of each sample to scale the values down
        exp_scores = np.exp(self.z2 - max_z)  # Subtract max from z2 to prevent overflow
        self.output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #self.output = sigmoid(self.z2)
        #self.output = (np.tanh(self.z2) + 1) / 2   #tanh orinal output in [-1,1], here scaled to [0,1]
        #self.output = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)  # softmax
        return self.output

def set_param(P, input_dim, hidden_dim, output_dim):

    P = np.array(P)
    nn = NeuralNetwork(input_dim, hidden_dim, output_dim)
    total_params = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim
    assert len(P) == total_params, "Parameter vector length does not match expected number of NN parameters."
    
    idx = 0
    nn.W1 = P[idx:idx+input_dim*hidden_dim].reshape(input_dim, hidden_dim)
    idx += input_dim*hidden_dim
    nn.b1 = P[idx:idx+hidden_dim].reshape(1, hidden_dim)
    idx += hidden_dim
    nn.W2 = P[idx:idx+hidden_dim*output_dim].reshape(hidden_dim, output_dim)
    idx += hidden_dim*output_dim
    nn.b2 = P[idx:idx+output_dim].reshape(1, output_dim)
    
    return nn

def get_output(indicators, nn):
    # Convert indicators to numpy array if not already
    x = np.array(indicators).reshape(1, -1)  # Reshape to (1, number_of_features)
    output = nn.forward(x)
    return output.flatten()  # Flatten in case output is multidimensional







class node_param:
    def __init__(self):
        self.c = []
        self.b = []
        self.w = []


#class ncRBF(object):

def get_outputs(inp, param, lin_param, N, M, K):
    # get layers charateristics
    # N = self.N # number of nodes in hidden layer
    # M = self.M # number of inputs
    # K = self.K # number of outputs

    phi = []
    o = []
    output = []

    for j in range(N):
        bf = 0

        for i in range(M):

            #logging.debug(f"Type of inp[{i}]: {type(inp[i])}, Value: {inp[i]}")
            #logging.debug(f"Type of param[{j}].c[{i}]: {type(param[j].c[i])}, Value: {param[j].c[i]}")
            #print(f"Type of inp[{i}]: {type(inp[i])}, Value: {inp[i]}")
            #print(f"Type of param[{j}].c[{i}]: {type(param[j].c[i])}, Value: {param[j].c[i]}")


            num = (inp[i] - param[j].c[i])*(inp[i] - param[j].c[i])
            den = (param[j].b[i]*param[j].b[i])

            if den < pow(10,-6):
                den = pow(10,-6)

            bf = bf + num / den

        phi.append( np.exp(-bf) )     # each element in the phi vector is the output of jth rbf in the hidden layer

    for k in range(K):
        o = lin_param[k]
        for j in range(N):
            o = o + param[j].w[k]*phi[j]   # compute the weighted sum of the outputs from the rbfs. param[j].w[k] is the weight of node j for output k


        if o > 1:
            o = 1.0
        if o < 0:
            o = 0.0     # make sure the output is in [0,1]

        output.append(o)

    return output


def set_params(policies, N, M, K):

    param_string = policies 
    count = 0
    lin_param = []
    param = []


    # linear parameters. As many as the outputs
    for k in range(K):
        lin_param.append(param_string[count])
        count += 1


    # RBF paramters
    for i in range(N): # nodes

        node = node_param()  #for each node,an instance of 'node_param' is created

        for j in range(M):
            node.c.append(param_string[count]) # center
            count += 1
            node.b.append(param_string[count]) # radius
            count += 1

        for k in range(K):
            node.w.append(param_string[count]) # output weight
            count += 1

        param.append(node)


    return param,lin_param

# set_param: P is K +  (2M + K) * N  length 
# The first K elements of P are used for linear parameters affecting the outputs directly.
#The rest of the P vector is divided among the N nodes, with each node receiving 2M + K parameters:
#The first M parameters for each node specify the centers of the RBF,
#The next M parameters specify the radii,
#The final K parameters specify the weights contributing to the node's output.