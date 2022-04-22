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
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from .processing import get_batch

"""
When given a gym environment and compatible policy maker the function 
collects a sample of date for a specified period and saves a replay.
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
            
            # monitor the sample size
            if len(memory) % (sample_size // 10) == 0 and len(memory) > 0: 
                print('Samples collected: {}'.format(len(memory)))
                if len(memory) == sample_size:
                    break
            
            # take an action and step the environment
            action = policy.get_action(state=state)
            next_state, reward, done, _ = env.step(action)  
            
            # add a termination penalty
            if done and timestep != kwargs.get("max_timestep", -1):
                reward += kwargs.get("termination_penalty", 0)
            
            # terminate if reached max timesteps
            if timestep == kwargs.get("max_timestep", -1):
                done = True
            
            # log the data
            sample = {"state": state, "next_state": next_state, 
                      "action": action, "done": done, "reward": reward}
            memory.append(sample)
            
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


"""
Trains a given online learning agent on a learned dynamics model 
"""        
def train_agent(env, policy, sample_size, offline_data, **kwargs):
    
    # set parameters
    device = kwargs.get("device", "cuda")
    memory = deque(maxlen=sample_size)
    
    # log the timesteps and reward
    episode_count = 0
    episode_reward, episode_timestep = [], []
        
    # data in episodes until memory is full    
    while len(memory) < sample_size:
        
        # sample a random starting state from the buffer  
        batch = get_batch(
            memory=offline_data,
            batch_size=1,
            device=device
        )        
        
        # reset the environmental parameters
        env.init_state = batch["state"].cpu().data.numpy()
        state = env.reset()        
        done, timestep, timestep, total_reward = False, 0, 0, 0
        
        # loop through the episode
        while not done:
            
            # monitor the sample size
            if len(memory) % (sample_size // 10) == 0 and len(memory) > 0: 
                print('Samples collected: {}'.format(len(memory)))
                if len(memory) == sample_size:
                    break
            
            # take an action and step the environment
            action = policy.get_action(state=state)            
            next_state, reward, done, _ = env.step(action)  
            
            # add a termination penalty
            if done and timestep != kwargs.get("max_timestep", -1):
                reward += kwargs.get("termination_penalty", 0)
            
            # terminate if reached max timesteps
            if timestep == kwargs.get("max_timestep", -1):
                done = True
            
            # log the data
            sample = {"state": state, "next_state": next_state, 
                      "action": action, "done": done, "reward": reward}
            memory.append(sample)
                                    
            # train the agent
            policy.train(replay_buffer=memory)
            
            # update the variables
            state = next_state
            policy.update()     
            total_reward += reward
            timestep += 1
            
        # update the logs
        episode_reward.append(total_reward)
        episode_timestep.append(timestep)
        episode_count += 1
        
        # Display the results
        freq = 10
        if episode_count % freq == 0:
            
            mean_reward = np.mean(episode_reward[-freq:])
            mean_timestep = np.mean(episode_timestep[-freq:])
            
            print('Ep: {:<5} - Reward: {:<7} - Timesteps: {:<5}'.format(episode_count, mean_reward, mean_timestep))
            
            if episode_count >= freq * 2:
                ep_arr = np.array(episode_reward)
                rolling_mean = np.mean(ep_arr.reshape(-1, freq), axis=1)
                plt.plot(list(range(len(rolling_mean))), rolling_mean)
                plt.show()
                
    # get the file name and path
    filepath = kwargs.get("filepath", "./") 
    filename = kwargs.get("filename", "training_weights")                
    policy.save(filepath + filename) 



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