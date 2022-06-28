#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:33:31 2022

@author: hemerson
"""

import numpy as np
import tensorflow as tf
import math, random, torch

"""
Add Gaussian Noise to the action output for 
improved exploration.
"""
def gaussian_class_wrapper(agent, scale=1):
    
    get_action = agent.get_action
        
    """
    Use variables stored within the agent class to
    keep track of previous noise values.
    """
    def get_action_wrapper(state):
        
        # get the action
        action = get_action(state)
        
        # get the gaussian noise
        mean, std = 0.0, 1.0
        if hasattr(agent, 'gaussian_mean'): mean = agent.gaussian_mean
        if hasattr(agent, 'gaussian_std'): std = agent.gaussian_std
        gaussian_noise = np.random.normal(loc=mean, scale=std, size=action.shape)        
        action += gaussian_noise * scale
        
        return action
    
    # update the method
    agent.get_action = get_action_wrapper
    
    return agent


"""
Add Ornstein-Uhlenbeck Noise to the action output for 
improved exploration.
"""
def ou_class_wrapper(agent, scale=1):
    
    get_action = agent.get_action
    agent.prev_ou_noise = 0
    sigma, theta, dt = 0, 0, 0
    
    """
    Use variables stored within the agent class to
    keep track of previous noise values.
    """
    def get_action_wrapper(state):
        
        # calculate the ou noise
        t1 = agent.prev_ou_noise
        t2 = theta * (0 - agent.prev_ou_noise) * dt
        t3 = sigma * math.sqrt(dt) * random.uniform(0, 1)
        ou_noise = t1 + t2 + t3
        ou_noise = ou_noise * scale
        agent.prev_ou_noise = ou_noise
        
        # get the action
        action = get_action(state)
        action += ou_noise
        
        return action
    
    # update the method
    agent.get_action = get_action_wrapper
    
    return agent


"""
Calculate the log porbability from the output of a deterministic agent.
"""
def get_log_prob(action, **kwargs):
    
    # get the parameters
    offset = kwargs.get("offset", 0)
    std = kwargs.get("std", 1)
    
    # get the mean and std using the normalised action
    mean = action + np.random.normal(loc=offset, scale=std, size=action.shape)
    
    # calculate the log prob
    norm_diff = (action - mean)/std
    log_prob = -(np.log(std) + 0.5*(np.log(2 * math.pi) + np.square(norm_diff)))
    
    return log_prob

"""
Convert the output of deterministic agent into a stochastic 
output in the form of log probability.
"""
def log_prob_class_wrapper(agent, **kwargs):
    
    get_action = agent.get_action
    
    """
    Add an additional output to the action.
    """
    def get_action_wrapper(state):
        
        action = get_action(state)
        log_prob = get_log_prob(action, **kwargs)        
        return action, log_prob
    
    # update the method
    agent.get_action = get_action_wrapper
    
    return agent

    
