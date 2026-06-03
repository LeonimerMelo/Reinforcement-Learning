# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:54:10 2025

@author: TechnoLEDs
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict

# 1. Configuração do Ambiente
env = gym.make('InvertedPendulum-v5', render_mode='human')  # Criando o ambiente

# 2. Parâmetros do Q-Learning
LEARNING_RATE = 0.1         # Taxa de aprendizado (alpha)
DISCOUNT_FACTOR = 0.95      # Fator de desconto (gamma)
EPISODES = 1000             # Número de episódios de treinamento
SHOW_EVERY = 500            # Mostrar renderização a cada X episódios
EPSILON = 1.0               # Taxa de exploração inicial
EPSILON_DECAY = 0.9995      # Decaimento da taxa de exploração
MIN_EPSILON = 0.01          # Taxa mínima de exploração

EPSILON = 1.0                 # Exploration probability at start          
MIN_EPSILON = 0.01           # Minimum exploration probability
epsilon_decay_rate = 5/EPISODES # Fator de decaimento do epsilon


# 3. Discretização do Espaço de Estados
# O ambiente tem espaços contínuos, então precisamos discretizá-los
NUM_BINS = 20               # Número de divisões para cada dimensão do estado

'''
def discretize_state(state):
    """Converte um estado contínuo em um estado discreto"""
    # O estado contém: [posição do carrinho, velocidade do carrinho, ângulo do pêndulo, velocidade angular]
    discretized = []
    for i in range(len(state)):
        # Normaliza cada componente do estado entre 0 e NUM_BINS-1
        scaled = np.interp(state[i], [env.observation_space.low[i], env.observation_space.high[i]], [0, NUM_BINS-1])
        discretized.append(int(round(scaled)))
    return tuple(discretized)
'''
# Limites manuais para o InvertedPendulum
STATE_BOUNDS = np.array([
    [-1.0, 1.0],    # posição do carrinho
    [-5.0, 5.0],    # velocidade do carrinho
    [-0.5, 0.5],    # ângulo
    [-10.0, 10.0]   # velocidade angular
])

def discretize_state(state):
    discretized = []

    for i in range(len(state)):
        low, high = STATE_BOUNDS[i]

        # clipping para evitar valores fora do intervalo
        value = np.clip(state[i], low, high)

        scaled = np.interp(value,
                           [low, high],
                           [0, NUM_BINS - 1])

        discretized.append(int(round(scaled)))

    return tuple(discretized)

# 4. Inicialização da Q-table
# Usamos um dicionário padrão que retorna 0 para estados/ações não vistos
# q_table = defaultdict(lambda: np.zeros(env.action_space.n))

ACTIONS = np.array([
    [-3.0],
    [-1.5],
    [0.0],
    [1.5],
    [3.0]
])

q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

# 5. Função para escolher ação usando política ε-greedy
def choose_action(state, epsilon):
    state_discrete = discretize_state(state)
    if np.random.random() < epsilon:
        # Exploração: escolhe ação aleatória
        # return env.action_space.sample()
        return np.random.randint(len(ACTIONS))
    else:
        # Explotação: escolhe melhor ação conhecida
        return np.argmax(q_table[state_discrete])

# 6. Treinamento
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Escolhe ação usando política ε-greedy
        action_idx = choose_action(state, EPSILON)
        action = ACTIONS[action_idx]
        
        # Executa ação no ambiente
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Discretiza os estados
        state_discrete = discretize_state(state)
        new_state_discrete = discretize_state(new_state)
        
        # Atualiza Q-value usando equação do Q-Learning
        # current_q = q_table[state_discrete][action]
        current_q = q_table[state_discrete][action_idx]
        max_future_q = np.max(q_table[new_state_discrete])
        
        # Fórmula do Q-Learning
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - current_q)
        # q_table[state_discrete][action] = new_q
        q_table[state_discrete][action_idx] = new_q
        
        state = new_state
        episode_reward += reward
        
        # Renderização opcional para visualização
        if episode % SHOW_EVERY == 0:
            env.render()
    
    # Decaimento da taxa de exploração
    # EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    EPSILON = max(EPSILON * (1 - epsilon_decay_rate), MIN_EPSILON)

    
    ep_rewards.append(episode_reward)
    
    # Estatísticas periódicas
    if not episode % 10:
        average_reward = sum(ep_rewards[-10:])/10
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-10:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-10:]))
        
        print(f"Episódio: {episode}, Recompensa média: {average_reward}, Recompensa min: {min(ep_rewards[-10:])}, Recompensa max: {max(ep_rewards[-10:])}, Epsilon: {EPSILON:.2f}")

env.close()

# 7. Visualização dos Resultados
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="Média")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="Mínima")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="Máxima")
plt.legend()
plt.title("Desempenho ao Longo dos Episódios")
plt.xlabel("Episódios")
plt.ylabel("Recompensa")
plt.show()