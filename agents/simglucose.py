#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math, random, gym
import matplotlib.pyplot as plt
from datetime import datetime
from gym.envs.registration import register

from .general import get_log_prob

# Register the child gym environment 
register(
    id='simglucose-child1-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'child#001'}
)

"""
A pid controller to control basal insulin dosing for child#1 
within the UVA/Padova Glucose Dynamics Simulator.
"""
class simglucose_pid:
    
    def __init__(self, **kwargs):
        
        # Child 1 Params
        self.bas = 1.142*(34.556/6000)*3
        self.target_blood_glucose = 144 
        self.integrated_state = 0
        self.previous_error = 0   
        
        # PID settings
        self.params = np.array([-1.290e-05, 2.067e-10, -1.103e-03])
        self.temp_params = self.params
        self.prev_reward = -500
        self.conserve_state = kwargs.get("conserve_state", True)
    
    """
    Reset the previous error and integrated state.
    """
    def reset(self):
        self.integrated_state = 0
        self.previous_error = 0           
    
    """
    Select the optimal dose of basal insulin to 
    keep blood glucose levels within range.
    """
    def get_action(self, state):
        
        # update for time-series decomposition
        current_bg = state[:, 0]        
        if len(state) < 11: current_bg += state[:, 1]           
        
        # proportional control
        error = self.target_blood_glucose - current_bg
        p_act = self.params[0] * error

        # integral control        
        self.integrated_state += error
        i_act = self.params[1] * self.integrated_state

        # derivative control
        d_act = self.params[2] * (error - self.previous_error)
        self.previous_error = error

        # get the final dose output
        action = np.array([(p_act + i_act + d_act + self.bas)/3], dtype=np.float32)
        
        # stops actions from seperate predictions affecting one another
        if not self.conserve_state:
            self.reset()        
    
        return np.maximum(action, 0.0)
    
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
        env = simglucose_class_wrapper(env)
        
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
Wrap the child#1 simglucose environment to modify the 
state, reward and add automatic bolus dosing.
"""        
def simglucose_class_wrapper(env, **kwargs): 
    
    # Set parameters
    horizon = kwargs.get("horizon", 1)
    use_condense_state = kwargs.get("use_condense_state", True)
    condense_state_type = kwargs.get("condense_state_type", "default")
    if use_condense_state: horizon = 80
    
    # Patient Parameters
    step, reset = env.step, env.reset
    cr, cf = 28.616, 103.017
    target_bg = 144
    
    # Track historical data
    env.logged_states = [] 
        
    """
    Wrap the env.step method to include automatic bolus dosing, the magni risk
    reward and a modified state including blood glucose, carbs ingested, total
    insulin dose and minute of the day.
    """
    def step_wrapper(basal_insulin):  
        
        # Include the bolus insulin dose -----------------------
        
        insulin_dose, bolus_insulin = np.copy(basal_insulin), 0
        current_meal, meals = 0, env.env.CHO_hist
        if len(meals) > 0:
            
            # extract the params from the simulator
            current_meal = meals[-1]
            blood_glucose = env.env.CGM_hist[-1]
            meal_sum = sum(env.env.CHO_hist[-60:])

            # calculate the meal bolus
            if current_meal > 0:
                
                # add estimation uncertainty
                current_meal += current_meal * random.uniform(-0.1, 0.1)                
                bolus_insulin += current_meal/cr
                if meal_sum == 0:
                     bolus_insulin += (blood_glucose - target_bg)/cf
                insulin_dose += bolus_insulin/3
        
        # step the environment
        blood_glucose, _, done, info = step(insulin_dose) 
                
        # Modify the outputs ------------------------------
        
        reward = magni_reward(blood_glucose)
        current_time = env.env.time_hist[-1]        
        time_in_mins = ((current_time.hour * 60) + current_time.minute)        
        state = np.array([blood_glucose[0], current_meal, insulin_dose, time_in_mins], dtype=float)   
                
        # Include historical data in state ---------------------------
        
        env.logged_states.insert(0, state)        
        padding_states = [env.logged_states[-1]] * max((horizon - len(env.logged_states)), 0)
        state = np.array(env.logged_states[:horizon] + padding_states, dtype=float).reshape(1, -1)
        
        # Condense the state ----------------------------------
        
        if use_condense_state:
            state, _ = condense_state(
                state=state,
                horizon=horizon,
                condense_state_type=condense_state_type
            )
            
        return state, reward, done, info   
    
    """
    Wrap the env.reset function to ensure that the first state is includes 
    blood glucose, carbohydrate, insulin and time information.    
    """
    def reset_wrapper():
        
        # get default insulin level
        patient_info = env.env.patient._params
        u2ss = patient_info['u2ss']
        BW = patient_info['BW']
        bas = u2ss * (BW/6000)
        
        # set state
        blood_glucose = reset()
        current_time = env.env.time_hist[-1]
        time_in_mins = ((current_time.hour * 60) + current_time.minute) 
        state = np.array([blood_glucose[0], 0, bas, time_in_mins])       
        
        # include historical data
        env.logged_states.insert(0, state)        
        padding_states = [env.logged_states[-1]] * max((horizon - len(env.logged_states)), 0)
        state = np.array(env.logged_states[:horizon] + padding_states, dtype=float).reshape(1, -1)
        
        # condense the state 
        if use_condense_state:
            state, _ = condense_state(
                state=state,
                horizon=horizon,
                condense_state_type=condense_state_type,
            )        
        
        return state
    
    # Define the new functions
    env.step = step_wrapper
    env.reset = reset_wrapper        
    
    return env


"""
A simple basal-bolus controller which just provides a continuous 
stream of basal insulin at a constant rate.
"""
class simglucose_basal:
    
    def __init__(self):
        # Child 1 Params
        self.bas = 1.142*(34.556/6000)*3
            
    def reset(self):
        pass
    
    def get_action(self, state):
        return np.array([self.bas/3])
    
    def update(self):
        pass

"""
Display the actions and achieved blood glucose values of 
agent evaluated on the simglucose environment. 
"""
def glucose_metrics(logs, window=480):  
    
    # check the logs all satisfy the window
    total_lengths = [min(len(log["state"]), window) for log in logs]
    error_message = "Input data does not span the specified window size: {}.".format(total_lengths)
    assert sum(total_lengths) / len(total_lengths) == total_lengths[0], error_message    
            
    # get the x-axis 
    x = list(range(total_lengths[0]))

    # Initialise the plot and specify the title
    fig = plt.figure(dpi=160)
    gs = fig.add_gridspec(4, hspace=0.0)
    axs = gs.subplots(sharex=True, sharey=False) 

    # define the hypo, eu and hyper regions
    axs[0].axhspan(180, 500, color='lightcoral', alpha=0.6, lw=0)
    axs[0].axhspan(70, 180, color='#c1efc1', alpha=1.0, lw=0)
    axs[0].axhspan(0, 70, color='lightcoral', alpha=0.6, lw=0)
    
    dose_max = 0
    for log in logs: 
        
        # unpackage the relevant metrics
        metrics = [(state[-1, 0], state[-1, 1], state[-1, 2]) for state in log["state"]] 
        blood_glucose, meals, insulin_doses = zip(*metrics)
        actions = [action[0] for action in log["action"]]                
        window = min(window, len(blood_glucose))
        blood_glucose, meals = np.array(blood_glucose[-window:]), np.array(meals[-window:])
        insulin_doses, actions = np.array(insulin_doses[-window:]), np.array(actions[-window:])
        
        # plot the values
        axs[0].plot(x, blood_glucose, label=log["name"])
        axs[1].plot(x, actions, label=log["name"])
        axs[2].plot(x, insulin_doses - actions, label=log["name"])
        axs[3].plot(x, meals, label=log["name"])
        dose_max = max(np.max(actions), dose_max)               

    # update the axis ranges    
    axs[0].legend(bbox_to_anchor=(1.0, 1.0))
    axs[0].axis(ymin=50, ymax=500)
    axs[0].axis(xmin=0.0, xmax=len(blood_glucose))
    axs[0].set_ylabel("BG \n(mg/dL)")
    axs[0].set_xlabel("Time \n(mins)")
    axs[1].axis(ymin=0.0, ymax=(dose_max * 1.4))
    axs[1].set_ylabel("Basal \n(U/min)")
    axs[2].axis(ymin=0.01, ymax=0.99)
    axs[2].set_ylabel("Bolus \n(U/min)")
    axs[3].axis(ymin=0, ymax=29.9)
    axs[3].set_ylabel("CHO \n(g/min)")

    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()

    plt.show()
    
#########################################################
# TODO: check conversion to numpy doesn't affect results
#########################################################
    
"""
Use the Magni risk function to calculate the reward for 
the current blood glucose value.
"""   
def magni_reward(blood_glucose):
    p1, p2, p3 = 3.5506, 0.8353, 3.7932
    reward = -10 * (p1 * (np.log(np.maximum(np.ones(1), blood_glucose[0]))**p2 - p3)) ** 2 
    return reward


"""
Transform the (horizon, 4) state into a condensed metric incorporating
all the important information.
"""
def condense_state(state, horizon=80, condense_state_type="default", **kwargs):
    
    # extract the relevant metrics
    state = state.reshape(-1, horizon, 4)       
    
    # convert to: (30-min bg over 4hrs, mob, iob, time)
    if condense_state_type == "default":        
        bg_intervals = state[:, list(range(0, horizon, horizon//8)) + [horizon - 1], 0].reshape(-1, 9)  
        mob = np.sum(state[:, :, 1] * np.flip(np.arange(horizon)/(horizon - 1)), axis=1).reshape(-1, 1)
        iob = np.sum(state[:, :, 2] * np.flip(np.arange(horizon)/(horizon - 1)), axis=1).reshape(-1, 1)
        current_time = state[:, 0, -1].reshape(-1, 1)
        trans_state = np.concatenate([bg_intervals, mob, iob, current_time], axis=1)
    
    # convert to: (season and trend data for bg, insulin, meals + time) 
    elif condense_state_type == "time_series_decomp": 
        trans_state = time_series_decomp(state, **kwargs)  
    
    # get the mean and standard deviation
    stats = [np.mean(trans_state, axis=0), np.std(trans_state, axis=0)]
    
    return trans_state, stats   


"""
Decompose time-series data into seasonality and trend
"""
def time_series_decomp(observations, **kwargs):
    
    # get parameters
    kernel_size = kwargs.get("kernel_size", 24)
    batch_size = observations.shape[0]
        
    # break into channels
    time_channel = observations[:, 0, -1]
    sensor_channel = observations[:, :, :-1]

    # get the trend data and seasonal data
    trend_channel = np.mean(sensor_channel[:, :kernel_size, :], axis=1)
    season_channel = sensor_channel[:, 0, :] - trend_channel    
    
    combined_channel = np.zeros((batch_size, 2*observations.shape[-1]-1))    
    for row in range(combined_channel.shape[-1]):        
        if (row == (combined_channel.shape[-1] - 1)): combined_channel[:, row] = time_channel
        elif row % 2 == 0: combined_channel[:, row] = season_channel[:, row//2]
        else: combined_channel[:, row] = trend_channel[:, (row-1)//2]

    return combined_channel


"""
Visualise the blood glucose predictions of a 
learned dynamics model.
"""
def display_glucose_prediction(true_values, pred_values):
    
    # unpackage the relevant metrics
    horizon = len(pred_values["state"])
    true_bg, pred_bg = true_values["state"][-horizon:, -1, 0], pred_values["state"][:, -1, 0]
    true_action, pred_action = true_values["action"][-horizon:], pred_values["action"]     
    
    # get the x-axis 
    x = list(range(len(true_bg)))

    # Initialise the plot
    fig = plt.figure(dpi=160)
    gs = fig.add_gridspec(2, hspace=0.0)
    axs = gs.subplots(sharex=True, sharey=False) 

    # define the hypo, eu and hyper regions
    axs[0].axhspan(180, 500, color='lightcoral', alpha=0.6, lw=0)
    axs[0].axhspan(70, 180, color='#c1efc1', alpha=1.0, lw=0)
    axs[0].axhspan(0, 70, color='lightcoral', alpha=0.6, lw=0)

    # plot the values
    axs[0].plot(x, true_bg, label="true")
    axs[0].plot(x, pred_bg, label="pred")
    axs[1].plot(x, true_action, label="true")    
    axs[1].plot(x, pred_action, label="pred")       

    # update the axis ranges    
    axs[0].legend(bbox_to_anchor=(1.0, 1.0))
    axs[0].axis(ymin=50, ymax=500)
    axs[0].axis(xmin=0.0, xmax=len(true_bg))
    axs[0].set_ylabel("BG \n(mg/dL)")
    axs[0].set_xlabel("Time \n(mins)")
    axs[1].axis(ymin=0.0, ymax=(max(true_action) * 1.4))
    axs[1].set_ylabel("Basal \n(U/min)")

    
    