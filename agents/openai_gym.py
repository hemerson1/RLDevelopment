#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import numpy as np


"""
A pid controller for solving the continuous lunar 
lander environment.
"""
class lunar_lander_pid:
    
    def __init__(self):        
        self.params = np.array([9.60095478, -7.12007871, -10.63690571, 9.60704393])      
        self.prev_reward = -300
        
    """
    Get the correct action to minimise the error.
    """
    def get_action(self, state):
        
        # Get the target values
        alt_tgt = abs(state[0])
        ang_tgt = (0.25 * np.pi) * (state[0] + state[2])

        # Calculate error values
        alt_error = (alt_tgt - state[1])
        ang_error = (ang_tgt - state[4])

        # Perform adjustments based on error
        alt_adj = self.params[0] * alt_error + self.params[1] * state[3]
        ang_adj = kp_ang = self.params[2] * ang_error + self.params[3] * state[5]

        # Clip the actions to the appropriate range
        action = np.array([alt_adj, ang_adj])
        action = np.clip(action, -1, +1)

        # Set action to zero if touching the ground
        if(state[6] or state[7]):
            action[:] = 0   
            
        return action
    
    """
    Determine the optimal parameters for the agent.
    """
    def optimise(self, episodes=100, start_exp=40.0):
        
        # Initialise the environment
        env = gym.make('LunarLanderContinuous-v2')
        self.params = np.zeros(4)
        env._max_episode_steps = 300
        
        for ep in range(1, episodes + 1):
            
            # add gaussian noise to the parameters
            self.temp_params = self.params
            self.params = self.params + np.random.normal(0, start_exp/ep, size=(4,))
            total_r = 0 
            
            # show learning update
            if ep % 10 == 0: print('Episode: {}'.format(ep))
            
            # test params on a sample of episodes
            for i in range(5):
                s, d = env.reset(), False               
                while not d:                    
                    a = self.get_action(s)
                    s, r, d, _ = env.step(a)
                    total_r += r
            
            # revert the params if not better
            if total_r/5 < self.prev_reward:
                self.params = self.temp_params
            else: self.prev_reward = total_r/5
        
        print('\n--------------------------------')
        print('Best params: {}'.format(self.params))
        print('Best reward: {}'.format(self.prev_reward))
        print('--------------------------------')
        
        """
        Update the agent.
        """        
        def update(self):
            pass
        