# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:15:17 2025

@author: Leonimer

Cliff Walking
=============
This environment is part of the Toy Text environments which contains general 
information about the environment.

Action Space: Discrete(4)
Observation Space: Discrete(48)
import:  gymnasium.make("CliffWalking-v0")

Cliff walking involves crossing a gridworld from start to goal while avoiding 
falling off a cliff.

Description
===========
The game starts with the player at location [3, 0] of the 4x12 grid world with 
the goal located at [3, 11]. If the player reaches the goal the episode ends.

A cliff runs along [3, 1..10]. If the player moves to a cliff location it returns 
to the start location. The player makes moves until they reach the goal.

Action Space
============
The action shape is (1,) in the range {0, 3} indicating which direction to move 
the player.

0: Move up
1: Move right
2: Move down
3: Move left

Observation Space
=================
There are 3 x 12 + 1 possible states. The player cannot be at the cliff, nor at 
the goal as the latter results in the end of the episode. What remains are all 
the positions of the first 3 rows plus the bottom-left cell.

The observation is a value representing the player’s current position as 
current_row * ncols + current_col (where both the row and col start at 0).

For example, the starting position can be calculated as follows: 3 * 12 + 0 = 36.
The observation is returned as an int().

Starting State
==============
The episode starts with the player in state [36] (location [3, 0]).

Reward
======
Each time step incurs -1 reward, unless the player stepped into the cliff, which 
incurs -100 reward.

Episode End
===========
The episode terminates when the player enters state [47] (location [3, 11]).

references
==========
https://gymnasium.farama.org/environments/toy_text/cliff_walking/
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pygame
# import pandas as pd

max_episode_steps_ = 500 # truncate afther this
env = gym.make('CliffWalking-v0', render_mode=None, 
               max_episode_steps = max_episode_steps_)

def epsilon_greedy_policy(Q, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Exploração
    else:
        return np.argmax(Q[state])  # Exploração baseada em Q

# Parâmetros
num_episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 0.1

steps_history = []
q_table_history = []
rewards_per_episode = []     # list to store rewards for each episode
Q = np.zeros((env.observation_space.n, env.action_space.n))  # Inicializa Q
# Treinamento
for episode in range(num_episodes):
    state, _ = env.reset()
    action = epsilon_greedy_policy(Q, state, env.action_space.n, epsilon)
    done = False
    rewards = 0
    steps = 0
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_action = epsilon_greedy_policy(Q, next_state, env.action_space.n, epsilon)

        # Atualiza Q usando a fórmula do SARSA
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state
        action = next_action

        rewards += reward
        steps += 1
        done =  terminated or truncated

    steps_history.append(steps)
    q_table_history.append(np.mean(Q))  # Armazenar média geral da Q-Table
    rewards_per_episode.append(rewards) # Store rewards per episode
    
    # if not solved: a partir de (episodes - 5) muda para render_mode='human'
    if episode == (num_episodes - 5):
        env.close()
        env = gym.make('CliffWalking-v0', render_mode='human')
        env.reset()

# encerra o ambiente de treinamento
env.reset()
env.close() 
pygame.quit()

# Função para imprimir o grid 4x12 com os índices das ações no grid 4x12
def print_optimal_values(q_table, rows, cols):
    optimal_values = np.argmax(q_table, axis=1)  # Seleciona o índice ótimo (argmax Q) para cada estado
    grid = optimal_values.reshape((rows, cols))  # Reshape para o formato do grid (4x12)

    print("índices de ações ótimas (argmax Q) no formato 4x12 grid:")
    for row in grid:
        print('-------------------------------------------------')
        print('|', " | ".join(f"{value:d}" for value in row), '|')
    print('-------------------------------------------------')

# 0: Move up
# 1: Move right
# 2: Move down
# 3: Move left
print('0: Move up')
print('1: Move right')
print('2: Move down')
print('3: Move left\n')

# Imprimir o grid com valores ótimos
print_optimal_values(Q, rows=4, cols=12)

# Graph mean rewards
mean_rewards_ = []
for t in range(num_episodes):
    # calculo a média móvel dos rewards de 100 episódios
    mean_rewards_.append(np.mean(rewards_per_episode[max(0, t-30):(t+1)]))
    
mean_steps_ = []
for t in range(num_episodes):
    # calculo a média móvel dos rewards de 100 episódios
    mean_steps_.append(np.mean(steps_history[max(0, t-30):(t+1)]))

plt.figure(figsize = (11,5))
plt.subplot(1,2,1)
plt.plot(q_table_history, label="Média Geral da Q-Table", color="blue")
plt.title("Evolução da Média Geral da Q-Table")
plt.xlabel("episodes")
plt.ylabel("Média dos Valores Q")
plt.xticks()
plt.yticks()
plt.grid()
plt.subplot(1,2,2)
plt.title('Mean rewards per episode')
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.plot(mean_rewards_, color="black")
plt.xticks()
plt.yticks()
plt.tight_layout()
plt.grid()
plt.show()

plt.plot(mean_steps_)
plt.title('Mean steps per episode')
plt.xlabel('episodes')
plt.ylabel('steps')
plt.show()