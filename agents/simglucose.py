#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math


"""
A pid controller to control basal insulin dosing for child#1 
within the UVA/Padova Glucose Dynamics Simulator.
"""
class simglucose_pid:
    
    def __init__(self):
        
        # Child 1 Params
        self.bas = 1.142 * (34.556/6000) * 3
        self.target_blood_glucose = 144 
        self.integrated_state = 0
        self.previous_error = 0   
        
        # PID settings
        self.params = np.array([-1.290e-05, 2.067e-10, -1.103e-03])
        self.temp_params = self.params
        self.prev_reward = -500
    
    """
    Select the optimal dose of basal insulin to 
    keep blood glucose levels within range.
    """
    def get_action(self, state):
        
        # proportional control
        error = self.target_blood_glucose - state[0] 
        p_act = self.params[0] * error

        # integral control        
        self.integrated_state += error
        i_act = self.params[1] * self.integrated_state

        # derivative control
        d_act = self.params[2] * (error - self.previous_error)
        self.previous_error = error

        # get the final dose output
        action = np.array([(p_act + i_act + d_act + self.bas) / 3], dtype=np.float32)
    
        return action
    
    """
    Determine the optimal parameters for the agent.
    """
    def optimise(self, episodes=100, start_exp=1.0):
        
        # Initialise the environment
        env = gym.make('simglucose-child1-v0')
        self.params = np.array([-1e-5, -1e-9, -1e-3])
        scale = np.array([1e-4, 1e-6, 1e-2])
        max_timesteps = 480
        
        # Wrap the environment
        env.step = bolus_class_wrapper(env)
        env.step = magni_class_wrapper(env)
        
        for ep in range(1, episodes + 1):
            
            # add gaussian noise to the parameters
            self.temp_params = self.params
            self.params = self.params + np.random.normal(0, start_exp/ep, size=(3,)) * scale
            
            # show learning update
            if ep % 10 == 0: print('Episode: {:03d} - Best Reward {}'.format(ep, self.prev_reward))
            
            # test params on a sample of episodes
            iters = 3
            total_r, total_t = 0, 0 
            for i in range(iters):
                s, d, t = env.reset(), False, 0               
                while not d:                        
                    a = self.get_action(s)
                    s, r, d, _ = env.step(a)
                    if t == (max_timesteps - 1): d = True
                    total_r += r
                    t += 1
                total_t += t 
                
            # revert the params if not better
            if total_r/iters < self.prev_reward or total_t/iters < max_timesteps:
                self.params = self.temp_params
            else: self.prev_reward = total_r/iters
        
        print('\n--------------------------------')
        print('Best params: {}'.format(self.params))
        print('Best reward: {}'.format(self.prev_reward))
        print('--------------------------------')
        
    """
    Update the agent.
    """        
    def update(self):
        pass
        
"""
Wrap the child#1 simglucose environment to automatically 
apply bolus insulin doses using pre-calculated parameters.
"""        
def bolus_class_wrapper(env):  
    
    func = env.step
    cr, cf = 28.616, 103.017
    target_bg = 144
    
    """
    Wrap the env.step method to include bolus insulin as 
    well as pre-calculated basal insulin,
    """
    def function_wrapper(basal_insulin):  
        
        meals, bolus_insulin = env.env.CHO_hist, 0
        if len(meals) > 0:
            
            # extract the params from the simulator
            current_meal = meals[-1]
            blood_glucose = env.env.CGM_hist[-1]
            meal_sum = sum(env.env.CHO_hist[-60:])

            # calculate the meal bolus
            if current_meal > 0:
                bolus_insulin += current_meal/cr
                if meal_sum == 0:
                     bolus_insulin += (blood_glucose - target_bg)/cf
        
        return func(basal_insulin + bolus_insulin/3)     
    
    return function_wrapper

"""
Use the magni risk function in place of the included reward function.
"""
def magni_class_wrapper(env):
    
    func = env.step
    p1, p2, p3 = 3.5506, 0.8353, 3.7932
    
    """
    Overwrite the reward output of the step function.
    """
    def function_wrapper(action):
        state, _, done, info = func(action)
        reward = -10 * (p2 * (math.log(max(1, state[0]))**p2 - p3)) ** 2   
        return state, reward, done, info
    
    return function_wrapper