# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:15:17 2025

@author: TechnoLEDs
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
num_episodes = 3
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
    if episode == (num_episodes - 3):
        env.close()
        env = gym.make('CliffWalking-v0', render_mode='human')
        env.reset()

# encerra o ambiente de treinamento
env.reset()
env.close() 
pygame.quit()

# Política aprendida
policy = np.argmax(Q, axis=1)
print("Política final (ações por estado):", policy)
print('\n')

# Exibir a Q-Table como DataFrame
# Criar um DataFrame para cada estado e ação
# def display_q_table(q_table):
#     rows = env.observation_space.n
#     cols = env.action_space.n
#     states = [(row, col) for row in range(rows) for col in range(cols)]
#     data = []
#     for state in states:
#         if Q[state[0], state[1]] == -1:  # Marcar obstáculos
#             data.append(["OBST", "OBST", "OBST", "OBST"])
#         else:
#             data.append(q_table[state].round(2))  # Valores Q para cada ação
#     df = pd.DataFrame(np.array(data), index=states, columns=['Up', 'Down', 'Left', 'Right'])
#     return df

# Exibir a Q-Table de forma organizada
def print_q_table(q_table):
    print("\nQ-Table:")
    for row in range(q_table.shape[0]):  # Para cada linha
        for col in range(q_table.shape[1]):  # Para cada coluna
            state = (row, col)
            # if grid[row, col] == -1:  # Obstáculo
            #     print(f"({row},{col}): OBST", end=" | ")
            # else:
            q_values = q_table[state]
            print(f"({row},{col}): {q_values.round(2)}", end=" | ")
        print()  # Nova linha para cada linha do grid

# Exibir a Q-Table
print_q_table(Q)


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