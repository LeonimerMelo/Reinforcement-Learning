# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:59:30 2025

@author: Leonimer
https://gymnasium.farama.org/environments/toy_text/blackjack/
https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/
"""

import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import pygame
from collections import defaultdict


env = gym.make("Blackjack-v1", sab=True, render_mode ='rgb_array')

# Whether to give an additional reward for starting with a natural blackjack, 
# i.e. starting with an ace and ten (sum is 21).
env = gym.make('Blackjack-v1', natural=True, sab=False, render_mode ='rgb_array')

# Whether to follow the exact rules outlined in the book by Sutton and Barto. 
# If `sab` is `True`, the keyword argument `natural`
env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode ='rgb_array')

# env = gym.make("Blackjack-v1", render_mode ='rgb_array')

'''
# Other possible environment configurations are:

env = gym.make('Blackjack-v1', natural=True, sab=False)
# Whether to give an additional reward for starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

env = gym.make('Blackjack-v1', natural=False, sab=False)
# Whether to follow the exact rules outlined in the book by Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
'''

state, info = env.reset()
print(state)
img = env.render()
plt.imshow(img)
plt.axis('off')
plt.show()

state_space = env.observation_space
print("There are", state_space, "possible states")
action_space = env.action_space.n
print("There are", action_space, "possible actions")

q_values = defaultdict(lambda: np.zeros(env.action_space.n))

# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1



state, info = env.reset()
print(state)
img = env.render()
plt.imshow(img)
plt.axis('off')
plt.show()


action = 0
action = 1
new_state, reward, terminated, truncated, info = env.step(action)

print(new_state, reward, terminated, truncated, info)
img = env.render()
plt.imshow(img)
plt.axis('off')
plt.show()


env = gym.make("Blackjack-v1", render_mode ='human')
tot_rew = 0
for epsode in range(400):
    state, info = env.reset()
    done = False
    step = 0
    rew=0
    while not done:
        # action = env.action_space.sample()
        action = 1
        if state[0] > 16:
            action = 0
            
        new_state, reward, terminated, truncated, info = env.step(action)
        step += 1
        
        state = new_state
        rew += reward
        print(epsode, step, action, new_state, reward, rew, terminated, truncated, info)

        done = terminated or truncated
    tot_rew += rew

print(tot_rew)
env.close()
