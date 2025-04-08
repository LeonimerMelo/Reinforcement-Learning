# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:57:48 2024

@author: Leonimer
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human')

state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
terminated = False      # True when fall in hole or reached goal
truncated = False       # True when actions > 200

while(not terminated and not truncated):
    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
    # action = 2
    new_state,reward,terminated,truncated,_ = env.step(action)
    state = new_state

env.close()

