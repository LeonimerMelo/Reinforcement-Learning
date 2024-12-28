# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 05:17:28 2024

@author: Leonimer

referências:
    https://gymnasium.farama.org/environments/toy_text/frozen_lake/
    
Frozen lake involves crossing a frozen lake from start to goal without 
falling into any holes by walking over the frozen lake. The player may not 
always move in the intended direction due to the slippery nature of the 
frozen lake.
"""

'''
Description
===========
The game starts with the player at location [0,0] of the frozen lake grid world 
with the goal located at far extent of the world e.g. [3,3] for the 
4x4 environment.

Holes in the ice are distributed in set locations when using a pre-determined 
map or in random locations when a random map is generated.

The player makes moves until they reach the goal or fall in a hole.

The lake is slippery (unless disabled) so the player may move perpendicular 
to the intended direction sometimes (see is_slippery).

Randomly generated worlds will always have a path to the goal.
'''

import numpy as np
import gymnasium as gym
#import gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
Modos de Renderização Disponíveis
human: Exibe o ambiente em uma janela interativa (requer suporte gráfico).
ansi: Imprime o estado do ambiente como texto (útil para terminais sem suporte gráfico).
rgb_array: Retorna uma matriz representando a imagem do ambiente (útil para gravações ou exibições personalizadas).

None (default): no render is computed.
human: render return None. The environment is continuously rendered in the current display or terminal. Usually for human consumption.
rgb_array: return a single frame representing the current state of the environment. A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y pixel image.
rgb_array_list: return a list of frames representing the states of the environment since the last reset. Each frame is a numpy.ndarray with shape (x, y, 3), as with rgb_array.
ansi: Return a strings (str) or StringIO.StringIO containing a terminal-style text representation for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
'''

# parâmetros do ambiente (env)
size_e = 4
seeds = 10
proba_frozen = .9
is_slip = True
max_episode_steps_ = 200

# Treinamento --> is_training = True
# Avaliação --> is_training = False
is_training = False
is_training = True

# if is training no render
render = True
if is_training:
    render = False

#env = gym.make("FrozenLake-v1", render_mode=None)
#env = gym.make("FrozenLake8x8-v1", render_mode=None)
# env = gym.make("FrozenLake-v1", render_mode=None, is_slippery=is_sl,
#         desc=generate_random_map(size=size_e, p=proba_frozen, seed=seeds))
env = gym.make("FrozenLake-v1", render_mode='human' if render else None,
               max_episode_steps = max_episode_steps_,
               is_slippery=is_slip, desc=generate_random_map(size=size_e, 
                                                           p=proba_frozen, 
                                                           seed=seeds))

# env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, 
#                max_episode_steps = max_episode_steps_,
#                render_mode='human' if render else None)

# Hiperparâmetros
episodes = 10000  # Número de episódios
alpha = 0.1  # Taxa de aprendizado (alpha or learning rate)
# gamma or discount rate. Near 0: more weight/reward placed on immediate state. 
# Near 1: more on future state
gamma = 0.9  # Fator de desconto  
epsilon = 1  # Taxa de exploração inicial (1 = 100% random actions)
#epsilon_decay_rate = 0.001  # Fator de decaimento do epsilon
min_epsilon = 0.01  # Epsilon mínimo
epsilon_decay_rate =  0.00001 # Fator de decaimento do epsilon

'''
Action Space
============
The action shape is (1,) in the range {0, 3} indicating which direction to move 
the player.

0: Move left
1: Move down
2: Move right
3: Move up

Observation Space
=================
The observation is a value representing the player’s current position as 
current_row * ncols + current_col (where both the row and col start at 0).

For example, the goal position in the 4x4 map can be calculated as 
follows: 3 * 4 + 3 = 15. The number of possible observations is dependent 
on the size of the map. The observation is returned as an int().

Starting State
==============
The episode starts with the player in state [0] (location [0, 0]).
'''
state_space = env.observation_space.n
action_space = env.action_space.n
# Inicializando a tabela Q
if is_training:
    q_table = np.zeros((state_space, action_space))
else:
    f = open('frozen_lake_'+str(size_e)+str(seeds)+'.pkl', 'rb')
    q_table = pickle.load(f)
    f.close()

# Função para escolher a ação (exploration vs. exploitation)
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # exploration
    else:
        return np.argmax(q_table[state])  # exploitation

'''
Rewards
=======
Reward schedule:
Reach goal: +1
Reach hole: 0
Reach frozen: 0

Episode End
===========
The episode ends if the following happens:
Termination:
The player moves into a hole.
The player reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, 
                                                                    max(ncol)-1]).
Truncation (when using the time_limit wrapper):
The length of the episode is 100 for 4x4 environment, 200 for 
FrozenLake8x8-v1 environment.
'''
# Treinamento
if is_training:
    # Monitorar métricas durante o treinamento
    rewards_per_episode = []
    steps_per_episode = []
    q_table_history = []
    #for episode in tqdm(range(episodes)):
    for episode in range(episodes):
        state = env.reset()[0] # reset state: 0 --> top left corner
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        while not done and not truncated:
            action = choose_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
    
            # Atualizando a tabela Q
            # best_next_action = np.argmax(q_table[next_state])
            
            # q_table[state, action] = q_table[state, action] + alpha * (
            #     reward + gamma * q_table[next_state, best_next_action] - q_table[state, action]
            # )
            
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            total_reward += reward
            steps += 1
    
        # Decaindo o epsilon
        a = epsilon * epsilon_decay_rate
        b = epsilon - a
        epsilon = max(b, min_epsilon)
        
        t = 100 * episode / episodes
        # print('episode: ', 'total: ', t, episode, 'epsilon: ', epsilon, end='\r')
        print('episode: %d \ttotal: %.1f%% \tepsilon: %.3f' % (episode, t, epsilon), end='\r')
        # epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
        # epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)
    
        # Log do progresso
        # if (episode + 1) % 100 == 0:
        #     print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        q_table_history.append(np.mean(q_table))  # Armazenar média geral da Q-Table
    
    f = open("frozen_lake_"+str(size_e)+str(seeds)+".pkl","wb")
    pickle.dump(q_table, f)
    f.close()
    print("\nTreinamento concluído!")
    
print("Tabela Q:")
print(q_table)

#env = gym.make("FrozenLake-v1", render_mode="human")
#env = gym.make("FrozenLake8x8-v1", render_mode="human")
# env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=is_sl,
#         desc=generate_random_map(size=size_e, p=proba_frozen, seed=seeds))

# render=True
# env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, 
#                render_mode='human' if render else None)

# Avaliação
if not is_training:
    state = env.reset()[0] # reset state: 0 --> top left corner
    done = False
    env.render()
    i=0
    rw=0
    truncated = False
    while not done and not truncated:
        action = np.argmax(q_table[state])
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        i+=1
        print('step time:', i, end='\r')
        rw+=reward
    
    print('\nreward:', rw)

env.close()

if is_training:
    a= 0
    for i in range(len(rewards_per_episode)):
        if rewards_per_episode[i] == 1.0:
            a+=1
            
    print('\nTotal + rewords:', a)
    print('Total - rewords:', len(rewards_per_episode) - a)
    
    # Métricas
    # Gráfico Total rewards per episode
    plt.plot(rewards_per_episode, '.')
    plt.xlabel('Episódios')
    plt.ylabel('Rewards')
    plt.title('Total rewards per episode')
    plt.show()
    
    # Gráfico Total steps per episode
    plt.plot(steps_per_episode, '.')
    plt.xlabel('Episódios')
    plt.ylabel('Steps time')
    plt.title('Total steps time per episode')
    plt.show()
    
    # Plotar a evolução da média geral da Q-Table
    plt.figure(figsize=(8, 6))
    plt.plot(q_table_history, label="Média Geral da Q-Table", color="blue")
    plt.title("Evolução da Média Geral da Q-Table")
    plt.xlabel("Episódios")
    plt.ylabel("Média dos Valores Q")
    plt.legend()
    plt.grid(True)
    plt.show()