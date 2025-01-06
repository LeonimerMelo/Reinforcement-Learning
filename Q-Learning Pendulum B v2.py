# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 10:57:40 2025

@author: Leonimer
"""

import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Treinamento --> is_training = True
# Avaliação --> is_training = False
is_training = False
is_training = True

avaliable_epsodes = 4 # episódios na avaliação do modelo

# if is training no render
render = True
if is_training:
    render = False
    
max_episode_steps_ = 500 # truncate afther this
# Criação do ambiente
env = gym.make('Pendulum-v1', render_mode='human' if render else None,
               max_episode_steps = max_episode_steps_)

# Hiperparâmetros
NUM_EPISODES = 1002
NUM_STATE_BINS = 15  # Número de bins para discretização do espaço de estados
NUM_ACTION_BINS = 10  # Número de bins para discretização do espaço de ações
LEARNING_RATE = 0.1  # Taxa de aprendizado
DISCOUNT_FACTOR = 0.99  # Fator de desconto
EPSILON = 1.0  # Probabilidade de exploração
EPSILON_DECAY = 3/NUM_EPISODES # epsilon decay rate
MIN_EPSILON = 0.05   # minimum epsilon
# EPSILON_DECAY = 0.995  # Decaimento de epsilon
# MIN_EPSILON = 0.1


# Discretização do espaço contínuo
def discretize(value, bins, min_value, max_value):
    """Mapeia um valor contínuo para um índice discreto."""
    return np.digitize(value, np.linspace(min_value, max_value, bins)) - 1
               
state_bounds = np.array([[-1.0, 1.0],  # cos(theta)
                         [-1.0, 1.0],  # sin(theta)
                         [-8.0, 8.0]])  # theta_dot
action_bounds = [-2.0, 2.0]  # Espaço de ações contínuo

# Inicialização da Q-table
q_table = np.zeros((NUM_STATE_BINS, NUM_STATE_BINS, NUM_STATE_BINS, NUM_ACTION_BINS))

# Política epsilon-greedy
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(NUM_ACTION_BINS)  # Ação aleatória
    return np.argmax(q_table[state])  # Melhor ação com base na Q-table

# treinamento
if is_training:
    q_table_history = []  
    rewards_per_episode = [] 
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        cos_theta, sin_theta, theta_dot = state
        state_indices = (
            discretize(cos_theta, NUM_STATE_BINS, *state_bounds[0]),
            discretize(sin_theta, NUM_STATE_BINS, *state_bounds[1]),
            discretize(theta_dot, NUM_STATE_BINS, *state_bounds[2]),
        )
        
        total_reward = 0
        done = False 
        truncated = False
        while not done and not truncated:
            # Seleciona ação
            action_idx = choose_action(state_indices, EPSILON)
            action = np.linspace(*action_bounds, NUM_ACTION_BINS)[action_idx]
            
            # Executa a ação no ambiente
            next_state, reward, done, truncated, _ = env.step([action])
            total_reward += reward
            
            # Atualiza o estado
            next_cos_theta, next_sin_theta, next_theta_dot = next_state
            next_state_indices = (
                discretize(next_cos_theta, NUM_STATE_BINS, *state_bounds[0]),
                discretize(next_sin_theta, NUM_STATE_BINS, *state_bounds[1]),
                discretize(next_theta_dot, NUM_STATE_BINS, *state_bounds[2]),
            )
            
            # Atualiza Q-table
            best_next_action = np.argmax(q_table[next_state_indices])
            q_table[state_indices][action_idx] += LEARNING_RATE * (
                reward
                + DISCOUNT_FACTOR * q_table[next_state_indices][best_next_action]
                - q_table[state_indices][action_idx]
            )
            
            # Avança para o próximo estado
            state_indices = next_state_indices
               
        rewards_per_episode.append(total_reward) # Store rewards per episode
        q_table_history.append(np.mean(q_table))  # Armazenar média geral da Q-Table
       
        # Decaindo o epsilon
        k = EPSILON - EPSILON * EPSILON_DECAY
        EPSILON = max(k, MIN_EPSILON) 
        # EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        
        print("\033[K", end="\r") # clear current print line
        print(f"Episode {episode + 1}/{NUM_EPISODES} Total Reward = {total_reward:.1f} epsilon = {EPSILON:.2f}", end='\r')

    # Save final Q-table to file
    f = open('pendulum_B_'+str(NUM_EPISODES)+'.pkl','wb')
    pickle.dump(q_table, f)
    f.close()
    
    # Plotar a evolução da média geral da Q-Table
    plt.figure(figsize=(8, 6))
    plt.plot(q_table_history, label="Média Geral da Q-Table", color="blue")
    plt.title("Evolução da Média Geral da Q-Table")
    plt.xlabel("Episódios")
    plt.ylabel("Média dos Valores Q")
    #plt.legend()
    plt.grid(True)
    plt.show()
    
    mean_rewards_ = []
    for t in range(NUM_EPISODES):
        # calculo a média móvel dos rewards de 100 episódios
        mean_rewards_.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.title('Mean rewards per episode')
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.plot(mean_rewards_)
    plt.show()

# Avaliação
else:
    tst = True
    while tst:
        try:
            # Read trained Q-table from file
            f = open('pendulum_B_'+str(NUM_EPISODES)+'.pkl', 'rb')
            q_table = pickle.load(f)
            f.close()
        except:
            print('arquivo não encontrado! Você deve treinar o modelo primeiro!')
            break
        
        env.reset()
        env.render()
        max_reward = -5000
        for eps in range(avaliable_epsodes):
            state, _ = env.reset()
            cos_theta, sin_theta, theta_dot = state
            state_indices = (
                discretize(cos_theta, NUM_STATE_BINS, *state_bounds[0]),
                discretize(sin_theta, NUM_STATE_BINS, *state_bounds[1]),
                discretize(theta_dot, NUM_STATE_BINS, *state_bounds[2]),
            )
            steps = 0
            done = False 
            truncated = False
            total_reward = 0
            while not done and not truncated:               
                # Seleciona ação
                action_idx = np.argmax(q_table[state_indices])  # ação com base na Q-table
                action = np.linspace(*action_bounds, NUM_ACTION_BINS)[action_idx]
                
                # Executa a ação no ambiente
                next_state, reward, done, truncated, _ = env.step([action])
                total_reward += reward
                
                # Atualiza o estado
                next_cos_theta, next_sin_theta, next_theta_dot = next_state
                next_state_indices = (
                    discretize(next_cos_theta, NUM_STATE_BINS, *state_bounds[0]),
                    discretize(next_sin_theta, NUM_STATE_BINS, *state_bounds[1]),
                    discretize(next_theta_dot, NUM_STATE_BINS, *state_bounds[2]),
                )
                
                # Avança para o próximo estado
                state_indices = next_state_indices
                steps += 1
                # max_reward = max(max_reward, rewards)
                t = 100 * (eps+1) / avaliable_epsodes
                print("\033[K", end="\r") # clear current print line
                print('episode: %d total: %.1f%% time steps: %d rewards: %d max reward: %d' % 
                      (eps+1, t, steps, total_reward, max_reward), end='\r') 
            max_reward = max(max_reward, total_reward)
             
        tst = False
        
# Encerramento do ambiente
env.close()
