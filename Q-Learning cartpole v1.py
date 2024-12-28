# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:18:40 2024

@author: TechnoLEDs
"""

import numpy as np
import gymnasium as gym
import random

# Criar o ambiente CartPole
#env = gym.make('CartPole-v1')
env = gym.make('CartPole-v1', render_mode="human")

# Definir parâmetros
alpha = 0.1  # Taxa de aprendizado
gamma = 0.99  # Fator de desconto
epsilon = 0.1  # Taxa de exploração
episodes = 100  # Número de episódios
max_steps = 200  # Número máximo de passos por episódio

# Discretizar o espaço de estados (aproximação)
n_bins = 10  # Número de bins para discretização dos estados
state_bins = [np.linspace(-x, x, n_bins) for x in [4.8, 5.0, 0.418, 5.0]]  # Limites do estado
q_table = np.zeros([len(state_bins[0]) + 1, len(state_bins[1]) + 1, len(state_bins[2]) + 1, len(state_bins[3]) + 1, 2])  # Tabela Q

# Função para discretizar o estado contínuo
def discretize_state(state):
    state_discretized = []
    for i in range(len(state)):
        state_discretized.append(np.digitize(state[0], state_bins[i]) - 1)
    return tuple(state_discretized)

# Política epsilon-greedy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploração
    else:
        r=np.argmax(q_table[state])
        if r > 1: r=1
        return r  # Exploração

# Treinamento
for episode in range(episodes):
    state = discretize_state(env.reset())  # Estado inicial
    total_reward = 0
    env.render()

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)  # Executar ação
        next_state = discretize_state(next_state)

        # Atualização da tabela Q
        q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

        state = next_state  # Atualizar o estado

        total_reward += reward  # Acumular recompensa

        if done:
            break  # Fim do episódio

    print(f'Episódio {episode + 1}/{episodes} - Recompensa: {total_reward}')

env.close()  # Close the environment when done

# Avaliação final
total_reward = 0
for _ in range(100):  # Testar o agente após o treinamento
    state = discretize_state(env.reset())
    for step in range(max_steps):
        r=np.argmax(q_table[state])
        if r>1:r=1
        action = r  # Escolher ação com maior valor Q
        next_state, reward, done, _, _ = env.step(action)
        state = discretize_state(next_state)
        total_reward += reward
        if done:
            break

print(f'Recompensa média após treinamento: {total_reward / 100}')