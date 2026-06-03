# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:40:57 2026

@author: TechnoLEDs
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# --- DEFINIÇÃO DO DISPOSITIVO (CUDA ou CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rodando no dispositivo: {device}")

# 1. Definição das Redes Neurais
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x): return self.net(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

# 2. Configurações e Inicialização
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0] # Correção para pegar o inteiro
action_dim = env.action_space.n

# Enviando as redes para a GPU (.to(device))
policy_net = PolicyNetwork(state_dim, action_dim).to(device)
value_net = ValueNetwork(state_dim).to(device)

policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
value_optimizer = optim.Adam(value_net.parameters(), lr=0.005)

GAMMA = 0.99
NUM_EPISODES = 500

history_rewards = []
history_policy_loss = []
history_value_loss = []

# 3. Loop Principal de Treinamento
for episode in range(NUM_EPISODES):
    state, info = env.reset()
    log_probs, state_values, rewards = [], [], []
    done = False
    
    while not done:
        # Transfere o estado atual para a GPU
        state_t = torch.FloatTensor(state).to(device)
        
        probs = policy_net(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        
        state_value = value_net(state_t)
        
        log_probs.append(dist.log_prob(action))
        state_values.append(state_value)
        
        # O ambiente roda na CPU, passamos a ação como tipo primitivo (.item())
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        state = next_state
        done = terminated or truncated
        
    # --- Fim do Episódio: Cálculos de Atualização ---
    discounted_returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        discounted_returns.insert(0, G)
        
    # Transfere todos os novos tensores calculados para a GPU
    returns_t = torch.FloatTensor(discounted_returns).to(device)
    log_probs_t = torch.stack(log_probs).to(device)
    state_values_t = torch.cat(state_values).squeeze(-1) # Já estava no device correto
    
    advantages = returns_t - state_values_t.detach()
    policy_loss = -(log_probs_t * advantages).mean()
    value_loss = nn.MSELoss()(state_values_t, returns_t)
    
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    # Salva métricas para o gráfico (.item() move do tensor CUDA para um float comum de CPU)
    total_reward = sum(rewards)
    history_rewards.append(total_reward)
    history_policy_loss.append(policy_loss.item())
    history_value_loss.append(value_loss.item())
    
    if episode % 20 == 0:
        print(f"Episódio {episode:03d} | Recompensa Total: {total_reward:.1f}")

env.close()
print("Treinamento concluído! Gerando gráficos...")

# --- PLOTAGEM DOS GRÁFICOS ---
plt.figure(figsize=(15, 5))

# Gráfico de Recompensas
plt.subplot(1, 2, 1)
plt.plot(history_rewards, label="Recompensa por Episódio", color="teal", alpha=0.6)
moving_avg = [np.mean(history_rewards[max(0, i-20):i+1]) for i in range(len(history_rewards))]
plt.plot(moving_avg, label="Média Móvel (20 ep)", color="darkblue", linewidth=2)
plt.title("Evolução dos Rewards")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Total")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

# Gráfico de Funções Loss
plt.subplot(1, 2, 2)
plt.plot(history_policy_loss, label="Loss do Ator (Política)", color="orangered", alpha=0.7)
plt.plot(history_value_loss, label="Loss do Crítico (Baseline)", color="purple", alpha=0.7)
plt.title("Evolução das Funções Loss")
plt.xlabel("Episódios")
plt.ylabel("Valor da Perda (Loss)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

'''
Gráfico de Rewards: A linha de tendência (média móvel) deve subir gradativamente. 
No CartPole-v1, o limite máximo de passos é 500. Quando a linha estabilizar próxima 
de 500, significa que o agente aprendeu perfeitamente a equilibrar o bastão.

Gráfico das Losses:
    A Loss do Crítico (Value Loss) começará alta (pois ele erra muito o 
valor de \(V(s)\) no início) e deve cair progressivamente em direção a zero conforme 
ele aprende a prever os retornos corretamente.
    A Loss do Ator (Policy Loss) pode flutuar bastante e até ficar negativa 
(já que é calculada com base no log da probabilidade ponderado pela vantagem). 
É normal ela subir um pouco no meio do treino enquanto o agente descobre caminhos 
melhores e depois estabilizar.
'''

'''
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 1. Definição das Redes Neurais
class PolicyNetwork(nn.Module):
    """Rede do Ator: mapeia estados para probabilidades de ações"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    """Rede do Crítico (Baseline): estima o valor esperado V(s)"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. Configurações e Inicialização
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)

policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
value_optimizer = optim.Adam(value_net.parameters(), lr=0.005)

GAMMA = 0.99
NUM_EPISODES = 500

# Listas para armazenar o histórico das métricas para os gráficos
history_rewards = []
history_policy_loss = []
history_value_loss = []

# 3. Loop Principal de Treinamento
for episode in range(NUM_EPISODES):
    state, info = env.reset()
    
    # Listas para armazenar a memória do episódio atual
    log_probs, state_values, rewards = [], [], []
    done = False
    while not done:
        state_t = torch.FloatTensor(state)
        
        # O Ator escolhe a ação baseada nas probabilidades
        probs = policy_net(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        
        # O Crítico calcula o valor estimado do estado atual V(s)
        state_value = value_net(state_t)
        
        # Armazena logs necessários para o cálculo do gradiente posterior
        log_probs.append(dist.log_prob(action))
        state_values.append(state_value)
        
        # Executa a ação no ambiente
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        
        state = next_state
        done = terminated or truncated
        
    # --- Fim do Episódio: Início do Cálculo de Atualização ---
    
    # 1. Calcula os retornos reais acumulados G_t (de trás para frente)
    discounted_returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        discounted_returns.insert(0, G)
        
    # Converte listas de dados coletados em Tensores do PyTorch
    returns_t = torch.FloatTensor(discounted_returns)
    log_probs_t = torch.stack(log_probs)
    state_values_t = torch.cat(state_values).squeeze(-1)
    
    # 2. Calcula a Vantagem (Retorno Real - Baseline)
    # Usamos o .detach() para o ator não afetar o aprendizado dos pesos do crítico
    advantages = returns_t - state_values_t.detach()
    
    # 3. Calcula as Funções de Perda (Loss)
    # Multiplicamos por -1 porque o PyTorch faz Descida de Gradiente, mas queremos Subida de Gradiente
    policy_loss = -(log_probs_t * advantages).mean()
    
    # Erro Quadrático Médio (MSE) para aproximar V(s) do retorno real G_t
    value_loss = nn.MSELoss()(state_values_t, returns_t)
    
    # 4. Atualização dos Parâmetros via Backpropagation
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    # Relatório de Progresso a cada 20 episódios
    total_reward = sum(rewards)
    # Salva as métricas do episódio atual no histórico
    history_rewards.append(total_reward)
    history_policy_loss.append(policy_loss.item())
    history_value_loss.append(value_loss.item())
    if episode % 20 == 0:
        print(f"Episódio {episode:03d} | Recompensa Total: {total_reward:.1f} | Loss Ator: {policy_loss.item():.4f} | Loss Crítico: {value_loss.item():.4f}")

env.close()
print("Treinamento concluído!")

print("Gerando gráficos...")

# --- CÓDIGO DOS GRÁFICOS ---
plt.figure(figsize=(15, 5))

# 1. Gráfico das Recompensas (Rewards)
plt.subplot(1, 2, 1)
plt.plot(history_rewards, label="Recompensa por Episódio", color="teal", alpha=0.6)
# Calcula uma média móvel de 20 episódios para suavizar a linha e ver a tendência
moving_avg = [np.mean(history_rewards[max(0, i-20):i+1]) for i in range(len(history_rewards))]
plt.plot(moving_avg, label="Média Móvel (20 ep)", color="darkblue", linewidth=2)
plt.title("Evolução dos Rewards ao Longo dos Episódios")
plt.xlabel("Episódios")
plt.ylabel("Recompensa Total")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

# 2. Gráfico das Funções Loss
plt.subplot(1, 2, 2)
plt.plot(history_policy_loss, label="Loss do Ator (Política)", color="orangered", alpha=0.7)
plt.plot(history_value_loss, label="Loss do Crítico (Baseline/MSE)", color="purple", alpha=0.7)
plt.title("Evolução das Funções Loss")
plt.xlabel("Episódios")
plt.ylabel("Valor da Perda (Loss)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
'''