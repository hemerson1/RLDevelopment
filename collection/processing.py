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
from collections import deque
import torch

"""
Given a sample of data this function unpacks the batch and normalises the 
state and action space. 

####################################
TODO: remove or clean this function 
####################################

"""
def clean_sample(memory, **kwargs):
        
    # set the parameters
    norm_reward = kwargs.get("norm_reward", False)
    reward_noise = kwargs.get("reward_noise", 0.0)
    standardise_dict = kwargs.get("standardise_dict", None)
    
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
        
    # cycle through the done array
    steps_log, init_states, steps = [], [], 0
    for idx, done in enumerate(done_array):
        
        # record initial states
        if steps == 0:
            init_states.append(state_array[idx, :])
        
        # record steps per trajectory
        steps += 1
        if done or idx == len(done_array) - 1:
            steps_log.append(steps)
            steps = 0            
    
    # get the number of trajectories and steps
    step_array = np.array(steps_log).reshape(-1, 1)
    init_state_array = np.array(init_states)
    num_trajectories = int(len(steps_log))
    
    # get the weights
    init_weight_array = np.random.multinomial(num_trajectories, [1.0 / num_trajectories] * num_trajectories, 1).reshape(-1, 1)
    weights = []
    for idx in range(num_trajectories):
        weights.append(np.ones(step_array[idx, :]) * init_weight_array[idx, :])        
    weight_array = np.concatenate(weights).reshape(-1, 1) 
    
    # calculate the mean and stds
    axis = tuple(range(state_array.ndim - 1))    
    state_mean, state_std = np.mean(state_array, axis=axis), np.std(state_array, axis=axis)
    action_mean, action_std = np.mean(action_array, axis=axis), np.std(action_array, axis=axis)
    reward_mean, reward_std = np.mean(reward_array, axis=axis), np.std(reward_array, axis=axis)
    
    # standardise the data 
    if standardise_dict is not None:    
        if "state" in standardise_dict:
            state_mean = standardise_dict["state"][:, 1]
            state_std = standardise_dict["state"][:, 0] - standardise_dict["state"][:, 1]        
        if "action" in standardise_dict:
            action_mean = standardise_dict["action"][:, 1]
            action_std = standardise_dict["action"][:, 0] - standardise_dict["action"][:, 1]
        if "reward" in standardise_dict:
            reward_mean = standardise_dict["reward"][:, 1]
            reward_std = standardise_dict["reward"][:, 0] - standardise_dict["reward"][:, 1]
    
    # add reward noise
    if reward_noise > 0: 
        reward_array[1::3, :] += reward_std * reward_noise
        reward_array[2::3, :] -= reward_std * reward_noise 
    
    # norm the data
    state_array = (state_array - state_mean) / (state_std + 1e-6)    
    next_state_array = (next_state_array - state_mean) / (state_std + 1e-6)
    init_state_array = (init_state_array - state_mean) / (state_std + 1e-6)
    action_array = (action_array - action_mean) / (action_std + 1e-6)
    if norm_reward: reward_array = (reward_array - reward_mean) / (reward_std + 1e-6)
    
    # calculate the max/min values
    state_max, state_min = np.max(state_array, axis=axis), np.min(state_array, axis=axis)
    action_max, action_min = np.max(action_array, axis=axis), np.min(action_array, axis=axis)
    reward_max, reward_min = np.max(reward_array, axis=axis), np.min(reward_array, axis=axis)    
    
    # package the statistics
    statistics = {
        "state": [state_mean, state_std, state_max, state_min], 
        "action": [action_mean, action_std, action_max, action_min], 
        "reward": [reward_mean, reward_std, reward_max, reward_min] 
        }
    
    # repackage the data
    clean_data = {
        
        # combined trajectories
        "state": state_array,    
        "next_state": next_state_array,   
        "action": action_array,   
        "reward": reward_array,   
        "done": done_array,  
        "weight": weight_array,
        
        # grouped by trajectory
        "step": step_array,
        "num_trajectory": num_trajectories,
        "init_weight": init_weight_array,
        "init_state": init_state_array        
        
        }
    
    return clean_data, statistics

"""
Given a replay return a batch of data from the replay of pre-specified size. 
"""
def get_batch(memory, batch_size, device="cpu", **kwargs):
    
    # set the parameters
    shuffle = kwargs.get("shuffle", True)    
    
    # initialise weight
    weight = None    
    
    # check if numpy array
    if type(memory) is dict:
        
        # generate a non-repeating set of indices
        full_idx = np.arange(memory["state"].shape[0])
        if shuffle: np.random.shuffle(full_idx)           
        chosen_idx = full_idx[:batch_size]
        
        num_elems = 5
        format_lambda = lambda x: torch.FloatTensor(x[chosen_idx, :].reshape(batch_size, -1)).to(device)   
        state, next_state, action, reward, done = list(map(format_lambda, list(memory.values())[:num_elems]))
                
    # check if deque
    elif type(memory) is deque:
        
        # get a random sample from the memory
        batch = random.sample(memory, batch_size) 
        
        format_lambda = lambda x: torch.FloatTensor(np.array([x]).reshape(batch_size, -1)).to(device) 
        unpacked_data = {key: [sample[key] for sample in batch] for key in batch[0]}       
        state, next_state, action, reward, done = map(format_lambda, unpacked_data.values())
    
    batch = {
        "state": state,    
        "next_state": next_state,   
        "action": action,
        "done": done,
        "reward": reward, 
        "weight": weight
        }
    
    return batch


"""
Split a provided numpy array of samples into a specified 
ratio of training and test data.
"""
def create_split(data, split):    
    
    train_data, test_data = {}, {}
    for key, val in data.items():    
        parts = np.split(val, [int(split * len(data["state"]))], axis=0)
        test_data[key] = parts[0]
        train_data[key] = parts[1]
        
    return train_data, test_data


"""
Convert the dataset to the appropriate form.
"""
def unpack_dataset(dataset, **kwargs):
        
    # reshape the input  
    data = dataset["trajectories"]
    state_dim, action_dim = np.array(data["observations"]).shape[-1], np.array(data["actions"]).shape[-1]
    data["observations"] = np.array(data["observations"], dtype=np.float32).reshape(-1, state_dim)
    data["next_observations"] = np.array(data["next_observations"], dtype=np.float32).reshape(-1, state_dim)
    data["actions"] = np.array(data["actions"], dtype=np.float32).reshape(-1, action_dim)
    data["rewards"] = np.array(data["rewards"], dtype=np.float32).reshape(-1, 1)
    data["terminals"] = np.array(data["terminals"], dtype=np.float32).reshape(-1, 1)
    
    # calculate the means and std
    stats = {
        "obs_mean": data["observations"].mean(axis=0), 
        "obs_std": data["observations"].std(axis=0), 
        "action_mean": data["actions"].mean(axis=0), 
        "action_std": data["actions"].std(axis=0), 
    }
    
    return data, stats

"""
Norm the state and action space.
"""
def norm_dataset(data, stats):
    data["observations"] = (data["observations"] - stats["obs_mean"])/stats["obs_std"]
    data["next_observations"] = (data["next_observations"] - stats["obs_mean"])/stats["obs_std"]
    data["actions"] = (data["actions"] - stats["action_mean"])/stats["action_std"]
    return data   


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
    
    
    
    
    
    
    
    
    
    
    
    
    
