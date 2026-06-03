# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:44:53 2025

@author: Leonimer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pygame

# Define a rede neural para a política
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        # Rede neural com uma camada oculta de 128 neurônios e ativação Softmax na saída
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  # Camada densa conectando entrada à camada oculta
            nn.ReLU(),                 # Função de ativação ReLU para não-linearidade
            nn.Linear(128, output_dim),  # Camada conectando a oculta à saída
            nn.Softmax(dim=-1)         # Softmax para gerar distribuição de probabilidade sobre ações
        )

    def forward(self, x):
        return self.fc(x)  # Passa o estado pela rede para obter as probabilidades das ações

# Cria o ambiente CartPole-v1
max_episode_steps_ = 3000
env = gym.make("CartPole-v1", render_mode=None, max_episode_steps = max_episode_steps_)

# Obtém o número de entradas (dimensão do estado) e saídas (número de ações)
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# Inicializa a política (rede neural) e o otimizador
policy = PolicyNetwork(obs_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.01)

print(policy) # print model pi
print(list(policy.parameters())[0].shape)
print(optimizer) # print optimizer parameters

loss_hist = []
rewards_hist = []
episodes = 600
gamma = 0.99
cont = 0
solved = False
episodes_run = 0

# Loop principal para treinar a política usando REINFORCE
for episode in range(episodes):  # Loop sobre os episódios
    state, _ = env.reset()  # Reinicia o ambiente e obtém o estado inicial
    log_probs = []  # Lista para armazenar os logaritmos das probabilidades
    rewards = []    # Lista para armazenar as recompensas
    discounted_returns = []  # Lista para armazenar os retornos descontados
    done = False  # Indica se o episódio terminou
    i = 0
    while not done:  # Enquanto o episódio não terminar
        # Converte o estado para um tensor para ser usado pela rede
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Obtém as probabilidades de ações a partir da política (softmax)
        action_probs = policy(state_tensor)
        # Define a distribuição categórica com base nas probabilidades
        action_dist = torch.distributions.Categorical(action_probs)
        # Seleciona uma ação amostrada da distribuição de acordo com as
        # probabilidades da saída da rede (softmax)
        action = action_dist.sample()
        # Armazena o log da probabilidade da ação tomada
        log_probs.append(action_dist.log_prob(action))
        # Executa a ação no ambiente e obtém o próximo estado e recompensa
        state, reward, terminated, truncated, _ = env.step(action.item())
        # Salva a recompensa recebida
        rewards.append(reward)
        # Termina o loop se o episódio estiver terminado ou truncado
        done = terminated or truncated

        if solved:
            i += 1
            print(i, '  ', end='\r')
       
    G = 0  # Inicializa o retorno acumulado
    # Calcula os retornos descontados para o episódio
    for reward in reversed(rewards):
        G = reward + gamma * G  # Calcula o retorno descontado
        discounted_returns.insert(0, G)  # Insere no início da lista
    # Converte os retornos para um tensor
    returns = torch.tensor(discounted_returns, dtype=torch.float32)
    # Normaliza os retornos para melhorar a estabilidade numérica
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    # policy gradient: Calcula a perda como -somatório dos log-probs ponderados pelos retornos
    # PyTorch torch.stack() method joins (concatenates) a sequence of tensors 
    # (two or more tensors) along a new dimension. 
    loss = -torch.sum(torch.stack(log_probs) * returns)
    loss_hist.append(loss.item())

    # Realiza a etapa de backpropagation
    optimizer.zero_grad()  # Zera os gradientes acumulados
    loss.backward()        # Calcula os gradientes da perda em relação aos parâmetros
    optimizer.step()       # Atualiza os pesos da rede com base nos gradientes

    episodes_run += 1
    # Calcula a recompensa total do episódio e exibe
    total_reward = int(sum(rewards))
    rewards_hist.append(total_reward)
    print("\033[K", end="\r") # clear current print line
    print(f"Episode {episode + 1}/{episodes}  \tLoss: {loss.item():.2f}  \tTotal Reward: {total_reward}", end='\r')

    # a partir de (episodes - 5) muda para render_mode='human'
    if episode == (episodes - 5):
        env.close()
        env = gym.make('CartPole-v1', render_mode='human', max_episode_steps = max_episode_steps_)
        env.reset()
      
    # Para o treinamento se o agente resolver o ambiente (alcançar reward_threshold)
    if total_reward >= env.spec.max_episode_steps:  
        print(f"\nSolved in {episode + 1} episodes!")
        solved = True 
        
    if solved :
        cont += 1    
        
    if cont == 1:
        env.close()
        env = gym.make('CartPole-v1', render_mode='human', max_episode_steps = max_episode_steps_)
        env.reset()  
        
    if cont > 3:
        break

# encerra o ambiente de treinamento
env.reset()
env.close() 
pygame.quit()

mean_loss_ = []
for t in range(episodes_run):
    # calculo a média móvel das perdas
    mean_loss_.append(np.mean(loss_hist[max(0, t-episodes_run//3):(t+1)]))
# Plotar o histórico da função perdas (loss)
plt.title("função perda (loss)")
plt.xlabel("Episódios")
plt.ylabel("loss")
plt.grid(True)
plt.plot(mean_loss_, color="red", label = 'mean loss')
plt.plot(loss_hist, label = 'episode loss')
plt.legend()
plt.show()

plt.title("Média Geral da função perda (loss)")
plt.xlabel("Episódios")
plt.ylabel("loss")
plt.grid(True)
plt.plot(mean_loss_, color="red")
plt.show()

mean_rewards_ = []
for t in range(episodes_run):
    # calculo a média móvel dos rewards
    mean_rewards_.append(np.mean(rewards_hist[max(0, t-episodes_run//6):(t+1)]))
# Plotar o histórico das recompensas
plt.title("Evolução das recompensas (rewards) por episódio")
plt.xlabel("Episódios")
plt.ylabel("rewards")
plt.grid(True)
plt.plot(mean_rewards_, label = 'mean rewards')
plt.plot(rewards_hist, color="red", label = 'rewards')
plt.legend()
plt.show()