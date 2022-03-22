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
import torch

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
    reward_array = np.array(unpacked_data["reward"]).reshape(-1, 1)
    done_array = np.array(unpacked_data["done"]).reshape(-1, 1)
    
    # convert to the correct number of dimensions
    if state_array.ndim < 2: 
        state_array = state_array.reshape(-1, 1)
        next_state_array = next_state_array.reshape(-1, 1)
    if action_array.ndim < 2:
        action_array = action_array.reshape(-1, 1)        
        
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

"""
Given a replay return a batch of data from the replay of pre-specified size. 
"""
def get_batch(memory, batch_size, device="cpu"):
    
    # generate a non-repeating set of indices
    full_idx = np.arange(memory["state"].shape[0])
    np.random.shuffle(full_idx)    
    chosen_idx = full_idx[:batch_size]
    
    # unpackage the memory and convert to tensor form
    state = torch.FloatTensor(memory["state"][chosen_idx, :]).to(device)
    action = torch.FloatTensor(memory["action"][chosen_idx, :]).to(device)
    next_state = torch.FloatTensor(memory["next_state"][chosen_idx, :]).to(device)
    reward = torch.FloatTensor(memory["reward"][chosen_idx, :]).to(device)
    done = torch.Tensor(memory["done"][chosen_idx, :]).to(device)
    
    batch = {
        "state": state,    
        "next_state": next_state,   
        "action": action,   
        "reward": reward,   
        "done": done,         
        }
    
    return batch


# TESTING -------------------------------------------------------------------
    
    
if __name__ == "__main__":
    
    from collections import deque

    # Test Data cleaning ---------------------------------    
    
    test_memory = deque(maxlen=100)
    for i in range(100):
        sample = {"state": [1, 1], "action": 1, "next_state": [1, 1], "reward": 2, "done": False}
        test_memory.append(sample)
    
    clean_memory, statistics = clean_sample(memory=test_memory)
    
    # Test batch retrieval -----------------------------
    
    batch = get_batch(memory=clean_memory, batch_size=5)
    
    # --------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    
    
    
