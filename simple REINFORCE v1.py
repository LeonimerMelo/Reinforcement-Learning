# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:18:45 2025

@author: TechnoLEDs
"""

import numpy as np
import gymnasium as gym

# Criar um ambiente (neste caso, CartPole)
env = gym.make('CartPole-v1')

# Parâmetros do modelo
learning_rate = 0.01
gamma = 0.99
num_episodes = 1

# Inicializar a rede neural (simplificada para este exemplo)
weights = np.random.rand(env.observation_space.shape[0], 1)
b = weights
# Função para escolher a ação
def policy(state):
    action_prob = np.dot(state, weights)
    action = 1 if np.random.rand() < action_prob else 0
    return action

# Treinamento
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    truncated = False
    while not done or not truncated:
        # Escolher ação
        action = policy(state)
        
        # Executar ação e observar o próximo estado e recompensa
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Calcular o gradiente e atualizar os pesos
        gradient = (reward + gamma * np.dot(next_state, weights)) * state
        a = learning_rate * gradient
        for i in range(len(a)):
            b[i] = a[i] + weights[i]
        weights = b
        # weights += learning_rate * gradient
        
        state = next_state
        total_reward += reward
        
        # print(gradient, weights)
        
    print(f"Episode: {episode}, Total reward: {total_reward}", end='\r')
    
env.close()