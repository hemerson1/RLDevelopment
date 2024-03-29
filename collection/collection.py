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
import pathos.multiprocessing as mp
import multiprocessing
import functools, itertools

from .processing import get_batch

"""
Run a single data collection episode.
"""
def run_episode(seed, env, policy, **kwargs):
    
    # abort the run if full
    if _abort_collect.is_set(): 
        return
    
    # initialise the data array    
    sequence_data = []
    log_obs, log_actions, log_next_obs = [], [], []
    log_terminals, log_rewards = [], []
    
    # reset the environmental parameters
    env.seed(seed)
    np.random.seed(seed)    
    
    state = env.reset()
    done, timestep, log_prob = False, 0, 0.0
    
    # loop through the episode
    while not done:

        # take an action 
        action_output = policy.get_action(state=np.copy(state))        
        if len(action_output) > 1: action, log_prob = action_output[0], action_output[1]
        else: action = action_output
        
        # step the environment
        next_state, reward, done, _ = env.step(action) 
        
        # add a termination penalty
        if done and timestep != kwargs.get("max_timestep", -1):
            reward += kwargs.get("termination_penalty", 0)

        # terminate if reached max timesteps
        if timestep == kwargs.get("max_timestep", -1):
            done = True
        
        # log the data
        log_obs.append(state)
        log_next_obs.append(next_state)
        log_actions.append(action)
        log_rewards.append(reward)
        log_terminals.append(done)

        # update the variables
        state = np.copy(next_state)
        policy.update()     
        timestep += 1
    
    # update the shared count
    with _counter_collect.get_lock():
        _counter_collect.value += len(log_obs)
        print('Collection: {}/{}'.format(_counter_collect.value, _sample_size))
        
        # Terminate training prematurely
        if _counter_collect.value >= _sample_size:
            print('Stopping worker early.')
            _abort_collect.set()
        
    # process the data into dictionary form
    sequence_data_dict = dict(
        observations=np.array(log_obs, dtype=np.float32).reshape(-1, log_obs[0].shape[-1]),
        actions=np.array(log_actions, dtype=np.float32).reshape(-1, log_actions[0].shape[-1]),
        next_observations=np.array(log_next_obs, dtype=np.float32).reshape(-1, log_obs[0].shape[-1]),
        rewards=np.array(log_rewards, dtype=np.float32).reshape(-1, 1),
        terminals=np.array(log_terminals, dtype=np.float32).reshape(-1, 1)            
    )      
        
    return sequence_data_dict


"""
Define the global variables for threaded
collection algorithm.
"""
def init_globals(counter, sample_size, abort):
    global _counter_collect
    global _sample_size
    global _abort_collect
    _sample_size = sample_size
    _counter_collect = counter
    _abort_collect = abort


"""
When given a gym environment and compatible policy maker the function 
collects a sample of date for a specified period and saves a replay.
"""
def collect_sample(env, policy, sample_size, **kwargs):
    
    # define the input function
    input_func = functools.partial(
        run_episode, env=env, 
        policy=policy, 
        **kwargs
    )
    
    # define the counter and abort events
    abort_event = multiprocessing.Event()
    counter_obj = multiprocessing.Value('i', 0)
    
    # create an iterator that terminates with pool
    def pool_args():
        for i in range(sample_size):
            if not abort_event.is_set():
                yield i    
                
    # define the pool
    pool = mp.Pool(
        kwargs.get("num_workers", 4), 
        initializer=init_globals, 
        initargs=(counter_obj, sample_size, abort_event)
    )  
    
    # run the multiprocessing
    list_output = pool.map(input_func, pool_args())
    dict_data = [item for item in list_output if item] 
    
    # cycle through the trajectories and append to a single list        
    memory_dict = dict(
      model_filename=kwargs.get("filename", "training_sample"),
      behavior_std=kwargs.get("std", 0.01),
      trajectories=dict(
          observations=[],
          actions=[],
          next_observations=[],
          rewards=[],
          terminals=[])
    )        
    for traj in dict_data:
        for k, v in traj.items():
            memory_dict['trajectories'][k].append(v)     
            
    # get the file name and path
    filepath = kwargs.get("filepath", "./") 
    filename = kwargs.get("filename", "training_sample")
    
    # save the memory as a pickle file
    with open(filepath + filename + '.pkl', 'wb') as file:
        pkl.dump(memory_dict, file)
        
    return memory_dict


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
                      "action": action, "reward": reward, "done": done}
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