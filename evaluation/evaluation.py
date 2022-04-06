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

"""
Given a policy and a gym environment this function runs a specified number of
test episodes.
"""
def test_policy(env, policy, episodes, **kwargs):
    
    test = False
    
    # run the policy for a set number of episodes
    episode_reward, episode_timestep = list(), list()
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
        episode_reward.append(total_reward)
        episode_timestep.append(timestep)
        
    # shut the window
    pygame.quit()
    
    # package results
    
    results = {
        "reward": episode_reward,
        "timestep": episode_timestep
        }
                
    return results


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
    
    
    