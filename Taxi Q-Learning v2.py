# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 16:18:57 2025

@author: Leonimer

https://thomassimonini.medium.com/q-learning-lets-create-an-autonomous-taxi-part-1-2-3e8f5e764358
"""

import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

env = gym.make("Taxi-v3", render_mode='rgb_array')

env.reset()
img = env.render()
plt.imshow(img)
plt.axis('off')
plt.show()

state_space = env.observation_space.n
print("There are", state_space, "possible states")
action_space = env.action_space.n
print("There are", action_space, "possible actions")

# Create our Q table with state_size rows and action_size columns (500x6)
Q = np.zeros((state_space, action_space))
print(Q)
print(Q.shape)

total_episodes = 30000         # Total number of training episodes
total_test_episodes = 10      # Total number of test episodes
max_steps = 200               # Max steps per episode

learning_rate = 0.01          # Learning rate
gamma = 0.99                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.001           # Minimum exploration probability
decay_rate = 0.0001             # Exponential decay rate for exploration prob

def epsilon_greedy_policy(Q, state):
  # if random number > greater than epsilon --> exploitation
  if(random.uniform(0,1) > epsilon):
    action = np.argmax(Q[state])
  # else --> exploration
  else:
    action = env.action_space.sample()

  return action

'''
Action Space The action shape is (1,) in the range {0, 5} indicating which direction 
to move the taxi or to pickup/drop off passengers.
0: Move south (down)
1: Move north (up)
2: Move east (right)
3: Move west (left)
4: Pickup passenger
5: Drop off passenger
'''

epsilon_hist = []
q_table_history = []
rewards_per_episode = []

for episode in tqdm(range(total_episodes)):
    # Reset the environment
    state, info_ = env.reset()
    step = 0
    done = False

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    epsilon_hist.append(epsilon)

    total_reward = 0
    for step in range(max_steps):
        #
        action = epsilon_greedy_policy(Q, state)

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, truncated, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q[state][action] = Q[state][action] + learning_rate * (reward + gamma *
                                    np.max(Q[new_state]) - Q[state][action])

        # If done : finish episode
        if done or truncated == True:
            break

        # Our new state is state
        state = new_state
        total_reward += reward

    q_table_history.append(np.mean(Q))  # Armazenar média geral da Q-Table
    rewards_per_episode.append(total_reward)

# save Q-Table
f = open("taxiv3.pkl","wb")
pickle.dump(Q, f)
f.close()
        
# Plotar o decaimento do epsilon ao longo dos epsódios
# plt.figure(figsize=(8, 6))
plt.plot(epsilon_hist, color="red")
plt.title("Decaimento do epsilon")
plt.xlabel("Episódios")
plt.ylabel("epsilon")
plt.grid(True)
plt.show()

# Plotar a evolução da média geral da Q-Table
# plt.figure(figsize=(8, 6))
xi = range(len(q_table_history))
plt.plot(xi, q_table_history, color="blue")
plt.title("Evolução da Média Geral da Q-Table")
plt.xlabel("Episódios")
plt.ylabel("Média dos Valores Q")
plt.grid(True)
plt.show()

# Gráfico Total rewards per episode
plt.plot(rewards_per_episode)
plt.xlabel('Rewards')
plt.ylabel('Total rewards')
plt.title('Total rewards per episode')
plt.show()

# rotina p/ cálculo da média móvel
def media_movel(dados, janela):
  media_movel_lista = []
  for i in range(len(dados) - janela + 1):
    media_movel = sum(dados[i:i+janela])/janela
    media_movel_lista.append(media_movel)
  return media_movel_lista

media_movel_ = media_movel(rewards_per_episode, janela=100)
plt.plot(media_movel_, color="black", alpha=0.9)
plt.title("Valor médio dos rewards")
plt.xlabel("Episódios")
plt.ylabel("média")
plt.grid(True)
plt.show()

# load Q-Table
f = open('taxiv3.pkl', 'rb')
Q = pickle.load(f)
f.close()

env = gym.make("Taxi-v3", render_mode='human')

for episode in range(15):
    state, info_ = env.reset()
    done = False
    truncated = False
    for step in range(max_steps):
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Q[state][:])
        new_state, reward, done, truncated, info = env.step(action)

        if done or truncated:
            break
        
        state = new_state


env.close()


# for i in range(100):
#     state, info_ = env.reset()
#     env.step(action)