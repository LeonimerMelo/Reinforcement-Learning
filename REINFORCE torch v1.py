# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:20:19 2025

@author: TechnoLEDs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
# import gym
# import numpy as np

# Definindo a rede neural para a política (actor)
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# REINFORCE Algorithm
def reinforce(env, policy_network, optimizer, episodes=1000, gamma=0.99):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        log_probs = []
        rewards = []

        while not done or not truncated:
            state = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_network(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            state, reward, done, truncated, _ = env.step(action.item())
            rewards.append(reward)

        # Calcular o retorno descontado (G_t)
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        # Normalizando os retornos para estabilidade
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Atualizar a política usando o gradiente
        loss = -torch.sum(torch.stack(log_probs) * returns)  # REINFORCE Update Rule
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if episode % 10 == 0:
        print(f"Episode {episode}/{episodes}, Loss: {loss.item()}", end='\r')

# Configuração do ambiente e do modelo
env = gym.make('CartPole-v1', render_mode=None)
# env = gym.make('CartPole-v1', render_mode='human')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

policy_network = PolicyNetwork(input_size, output_size)
optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)

# Treinando o agente com REINFORCE
reinforce(env, policy_network, optimizer, episodes=100)

# env.reset()
# env = gym.make('CartPole-v1', render_mode='human')

env.close()
