#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:33:37 2022

@author: hemerson
"""

"""
Functions for collecting data from gym environments using a pre-specified
policy maker
"""

from collections import deque
import pickle as pkl


"""
When given a gym environment and compatible policy maker the function 
creates returns and saves a replay of specified size.
"""

def collect_sample(env, policy, sample_size, **kwargs):
    
    # initialise the memory
    memory = deque(maxlen=sample_size)
    
    # data in episodes until memory is full    
    while len(memory) < sample_size:
        
        # reset the environmental parameters
        state = env.reset()
        done, timestep = False, 0
        
        # loop through the episode
        while not done:
            
            # take an action and step the environment
            action = policy.get_action(state=state)
            next_state, reward, done, _ = env.step(action)            
            
            # log the data
            sample = {"state": state, "next_state": next_state, 
                      "action": action, "done": done, "reward": reward}
            memory.append(sample)
            
            # terminate if reached max timesteps
            if timestep == kwargs.get("max_timestep", -1):
                done = True
            
            # update the variables
            state = next_state
            policy.update()     
            timestep += 1
    
    # get the file name and path
    filepath = kwargs.get("filepath", "./") 
    filename = kwargs.get("filename", "training_sample")
    
    # save the memory as a pickle file
    with open(filepath + filename + '.pkl', 'wb') as file:
        pkl.dump(memory, file)
        
    return memory    



# TESTING -------------------------------------------------------------------
    
if __name__ == "__main__":
    
    import gym
    
    # create a test policy
    class test_agent:
        
        def __init__(self):
            pass
        
        def get_action(self, state):
            return 0            
        
        def update(self):
            pass
        
    # create a text environment
    env = gym.make("CartPole-v1")    
    
    # instantiate the data collection
    data = collect_sample(
        env=env, 
        policy=test_agent(),
        sample_size=1000,
        max_timestep=2,
        filepath="./RLDevelopment/",
        filename="test"
        )