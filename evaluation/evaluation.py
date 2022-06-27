#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:34:17 2022

@author: hemerson
"""

"""
Functions for testing the learned policies of offline RL agents 
"""

import pygame, tqdm
import numpy as np
import pathos.multiprocessing as mp
import multiprocessing
import functools

"""
Given a policy and a gym environment this function runs a specified number of
test episodes.
"""
def test_policy(seed, env, policy, **kwargs):
    
    import torch
        
    # reset the environmental parameters
    env.seed(seed)
    state = env.reset()
    timestep, total_reward, done = 0, 0, False 

    # loop through the episode
    while not done:

        # take an action
        action_output = policy.get_action(state=state)  
        if len(action_output) > 1: action, log_prob = action_output[0], action_output[1]
        else: action = action_output
        
        # step the environment
        next_state, reward, done, _ = env.step(action)   

        # display the environment
        if kwargs.get("render", False):
            env.render()

        # terminate if reached max timesteps
        if timestep == kwargs.get("max_timestep", -1):
            done = True

        # update the variables
        state = next_state
        policy.update()     
        timestep += 1
        
        # tally the reward
        if kwargs.get("apply_discount", False): 
            reward = reward * kwargs.get("discount", 0.99) ** timestep
        total_reward += reward           
        
    # shut the window
    pygame.quit()
                
    return total_reward, timestep


"""
Compare the predictions of a given dynamics model against the true 
environmental outcomes over a specified horizon.
"""
def test_prediction(env, env_model, policy, horizon, **kwargs): 
    
    # specify the starting timestep
    starting_timestep = kwargs.get("starting_timestep", 0)
    horizon = horizon + starting_timestep - 1
        
    # get the true outcome ---------------------------------
    
    # reset the environmental parameters
    state, timestep, done = env.reset(), 0, False 
    true_states, true_actions, true_rewards, true_dones = [], [], [], [] 

    # loop through the episode
    while not done:
        
        # log the starting timestep
        if timestep == starting_timestep:
            init_state = state

        # take an action and step the environment
        action = policy.get_action(state=state)
        next_state, reward, done, _ = env.step(action)   

        # terminate if reached max timesteps
        if timestep == horizon:
            done = True
            
        # update the variables
        state = next_state
        policy.update()     
        timestep += 1

        # update the logs
        true_states.append(state)
        true_actions.append(action)
        true_dones.append(done)
        true_rewards.append(reward)        
        
    # test the prediction --------------------------------
    
    # reset the environmental parameters
    pred_states, pred_actions, pred_rewards, pred_dones = [], [], [], []
    env_model.init_state = init_state
    state, done, timestep = env_model.reset(), False, 0
    
    while not done:
        
        # take an action and step the environment
        action = policy.get_action(state=state)        
        next_state, reward, done, _ = env_model.step(action)   

        # terminate if reached max timesteps
        if timestep == (horizon - starting_timestep):
            done = True

        # update the variables
        state = next_state
        policy.update()     
        timestep += 1

        # update the logs
        pred_states.append(state)
        pred_actions.append(action)
        pred_dones.append(done)
        pred_rewards.append(reward)  
        
    # get the results ------------------------------------
    
    # package logged data
    true_values = {
        "state": np.array(true_states),
        "action": np.array(true_actions),
        "done": np.array(true_dones),
        "reward": np.array(true_rewards)
    }
        
    pred_values = {
        "state": np.array(pred_states),
        "action": np.array(pred_actions),
        "done": np.array(pred_dones),
        "reward": np.array(pred_rewards)
    }
    
    return true_values, pred_values

"""
Get an estimate of the return of a policy 
on the simglucose environment.
"""
def get_monte_carlo_return(env, policy, num_runs, **kwargs):
    
    # define the input function
    input_func = functools.partial(
        test_policy, env=env, 
        policy=policy, 
        **kwargs
    )
    
    # run with more workers
    if kwargs.get("num_workers", 1) > 1:
    
        # define the pool and run the multiprocessing
        print('Workers Started Evaluating.')
        pool = mp.Pool(kwargs.get("num_workers", 4))  
        list_output = list(tqdm.tqdm(pool.imap(input_func, range(num_runs)), total=num_runs))
        unpacked_list = [item[0] for item in list_output]
        print('Workers Finished.')
        
    else:
        unpacked_list = []
        for s in range(num_runs):
            unpacked_list.append(input_func(s)[0])            
    
    # sum the rewards
    reward_sum = np.sum(unpacked_list)
    normed_reward = reward_sum/num_runs * (1 - kwargs.get("discount", 0.99))
    
    return normed_reward

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
    results = test_policy(
        env=env, 
        policy=test_agent(),
        episodes=10,
        render=False
        )
    
    print(results["reward"])
    
    
    