#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:34:17 2022

@author: hemerson
"""

"""
Functions for testing the learned policies of offline RL agents 
"""

import pygame
import numpy as np

"""
Given a policy and a gym environment this function runs a specified number of
test episodes.
"""
def test_policy(env, policy, episodes, **kwargs):
    
    # run the policy for a set number of episodes
    episode_reward, episode_timestep = [], []
    states, actions, rewards, dones = [], [], [], []
    
    for ep in range(episodes): 
        
        # reset the environmental parameters
        state = env.reset()
        timestep, total_reward, done = 0, 0, False 
        
        # loop through the episode
        while not done:
            
            # take an action and step the environment
            action = policy.get_action(state=state)
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
            total_reward += reward   
            
            # update the logs
            states.append(state)
            actions.append(action)
            dones.append(done)
            rewards.append(reward)
            
        # update the logs
        episode_reward.append(total_reward)
        episode_timestep.append(timestep)
        
    # shut the window
    pygame.quit()
    
    # package results    
    results = {
        "reward": episode_reward,
        "timestep": episode_timestep
        }
    
    # package logged data
    log = {
        "state": states,
        "action": actions,
        "done": dones,
        "reward": rewards,
        "episodes": episodes
    }
                
    return results, log


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
    state, done, timestep = env_model.reset(init_state), False, 0
    
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
    
    
    