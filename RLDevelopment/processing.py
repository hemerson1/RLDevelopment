#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:34:55 2022

@author: hemerson
"""

"""
Functions for processing data before training and loading into offline RL 
algorithms.
"""

import numpy as np
import random

"""
Given a sample of data this function unpacks the batch and normalises the 
state and action space. 
"""
def clean_sample(memory):
    
    # randomly shuffle the data
    memory = random.sample(list(memory), len(memory))
    
    # package the data together
    unpacked_data = {key: [sample[key] for sample in memory] for key in memory[0]}
    state_array = np.array(unpacked_data["state"])
    next_state_array = np.array(unpacked_data["next_state"])
    action_array = np.array(unpacked_data["action"])
    reward_array = np.array(unpacked_data["reward"])
    done_array = np.array(unpacked_data["done"])    
        
    # calculate the mean and stds
    state_mean, state_std = np.mean(state_array, axis=0), np.std(state_array, axis=0)
    action_mean, action_std = np.mean(action_array, axis=0), np.std(action_array, axis=0)
    
    # norm the data
    state_array = (state_array - state_mean) / (state_std + 1e-6)
    next_state_array = (next_state_array - state_mean) / (state_std + 1e-6)
    action_array = (action_array - action_mean) / (action_std + 1e-6)
    
    # package the statistics
    statistics = {
        "state": [state_mean, state_std], 
        "action": [action_mean, action_std], 
        }
    
    # repackage the data
    clean_data = {
        "state": state_array,    
        "next_state": next_state_array,   
        "action": action_array,   
        "reward": reward_array,   
        "done": done_array,   
        }
    
    return clean_data, statistics
    
    
if __name__ == "__main__":
    
    from collections import deque
    
    
    test_memory = deque(maxlen=100)
    for i in range(100):
        sample = {"state": 1, "action": 1, "next_state": 1, "reward": 2, "done": False}
        test_memory.append(sample)
    
    clean_memory, statistics = clean_sample(memory=test_memory)
    
    print(clean_memory)
    print(statistics)
