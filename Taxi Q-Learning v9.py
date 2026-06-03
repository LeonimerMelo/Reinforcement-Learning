# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 16:18:57 2025

@author: Leonimer
https://gymnasium.farama.org/environments/toy_text/taxi/

Taxi
====
This environment is part of the Toy Text environments which contains general 
information about the environment.

Action Space = Discrete(6)
Observation Space = Discrete(500)

The Taxi Problem involves navigating to passengers in a grid world, picking them up 
and dropping them off at one of four locations.

Description
===========
There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) 
in the 5x5 grid world. The taxi starts off at a random square and the passenger at one 
of the designated locations.
The goal is move the taxi to the passenger’s location, pick up the passenger, move to 
the passenger’s desired destination, and drop off the passenger. Once the passenger 
is dropped off, the episode ends.
The player receives positive rewards for successfully dropping-off the passenger 
at the correct location. Negative rewards for incorrect attempts to pick-up/drop-off 
passenger and for each step where another reward is not received.

Map:
    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+

Observation Space
=================
There are 500 discrete states since there are 25 taxi positions, 5 possible locations 
of the passenger (including the case when the passenger is in the taxi), and 4 
destination locations.
Destination on the map are represented with the first letter of the color.

Passenger locations:
0: Red
1: Green
2: Yellow
3: Blue
4: In taxi

Destinations:
0: Red
1: Green
2: Yellow
3: Blue

An observation is returned as an int() that encodes the corresponding state, 
calculated by ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

Note that there are 400 states that can actually be reached during an episode. 
The missing states correspond to situations in which the passenger is at the same 
location as their destination, as this typically signals the end of an episode. 
Four additional states can be observed right after a successful episodes, when both 
the passenger and the taxi are at the destination. This gives a total of 404 reachable 
discrete states.
"""

import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import pygame
# import ale_py
# import shimmy

# Treinamento --> is_training = True
# Avaliação --> is_training = False
is_training = False
# is_training = True

max_episode_steps_ = 200
# estruturando o ambiente. if is training no render
env = gym.make("Taxi-v4", render_mode ='rgb_array' if is_training else 'human',
               max_episode_steps = max_episode_steps_)

if is_training:
    # take a look at grid
    state, info = env.reset()
    img = env.render()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

state_space = env.observation_space.n
print("There are", state_space, "possible states")
action_space = env.action_space.n
print("There are", action_space, "possible actions")

# hiperâmetros
total_episodes = 4000         # Total number of training episodes
learning_rate = 0.06          # Learning rate
gamma = 0.9                  # Discounting rate
# Exploration parameters
epsilon = 1.0                 # Exploration probability at start          
min_epsilon = 0.01           # Minimum exploration probability
epsilon_decay_rate = 5/total_episodes # Fator de decaimento do epsilon

if is_training:
    # Create our Q table with state_size rows and action_size columns (500x6)
    Q = np.zeros((state_space, action_space))
else:
    print('Carregando o modelo pré-treinado....')
    f = open('taxi_'+str(total_episodes)+'.pkl', 'rb')
    Q = pickle.load(f)
    f.close()

print(Q)
print(Q.shape)

def epsilon_greedy_policy(Q, state):
  # if random number > greater than epsilon --> exploitation
  if(random.uniform(0,1) > epsilon):
    action = np.argmax(Q[state])
  # else --> exploration
  else:
    action = env.action_space.sample()
  return action

'''
Action Space The action shape is (1,) in the range {0, 5} indicating which direction 
to move the taxi or to pickup/drop off passengers.
0: Move south (down)
1: Move north (up)
2: Move east (right)
3: Move west (left)
4: Pickup passenger
5: Drop off passenger

Rewards
-1 per step unless other reward is triggered.
+20 delivering passenger.
-10 executing “pickup” and “drop-off” actions illegally.

An action that results a noop, like moving into a wall, will incur the time step penalty. 
Noops can be avoided by sampling the action_mask returned in info.
'''

# Treinamento
if is_training:
    epsilon_hist = []
    q_table_history = []
    rewards_per_episode = []
    
    for episode in tqdm(range(total_episodes)):
        # Reset the environment
        state, info_ = env.reset()
        step = 0
        done = False
        total_reward = 0
        while not done:
            action = epsilon_greedy_policy(Q, state)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Q[state][action] = Q[state][action] + learning_rate * (reward + gamma *
                                        np.max(Q[new_state]) - Q[state][action])
            # If done : finish episode
            if terminated or truncated:
                done = True
            # Our new state is now state
            state = new_state
            total_reward += reward

        # Reduce epsilon (because we need less and less exploration)
        epsilon = max(epsilon * (1 - epsilon_decay_rate), min_epsilon)
        # epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        epsilon_hist.append(epsilon)

        q_table_history.append(np.mean(Q))  # Armazenar média geral da Q-Table
        rewards_per_episode.append(total_reward)

    f = open('taxi_'+str(total_episodes)+'.pkl','wb')
    pickle.dump(Q, f)
    f.close()
    print("\nTreinamento concluído!")

print(Q)

# Avaliação
nr_episodes = 10
if not is_training:
    act = ['down', 'up', 'right', 'left', 'pick', 'drop']
    for i in range(nr_episodes):
        state, info = env.reset() # reset state
        done = False
        step=0
        cont=0
        act_flag = False
        while not done:
            if act_flag:
                # Noops can be avoided by sampling the action_mask returned in inf
                # o info["action_mask"] me mostra as ações (=1) possíveis de se 
                # efetuar no presente estado do agente
                print('info action')
                action = np.argmax(info["action_mask"])
                act_flag = False
            else:
                action = np.argmax(Q[state])
            
            new_state, reward, terminated, truncated, info = env.step(action)
            
            # se ficar empacado ativo a leitura do info["action_mask"]!!
            if state == new_state:
                cont += 1
                if cont > 7:
                    act_flag = True
                    cont = 0
            else:
                cont = 0
                    
            print(f"epsode: {i+1} step: {step:<5} action: {act[action]:<8} reward: {reward} info: {info["action_mask"]} new state: {new_state}")
            state = new_state
            step += 1
            if terminated or truncated:
                done = True

env.close()
pygame.quit()

# Metrics after training
if is_training:
    # Plotar o decaimento do epsilon ao longo dos epsódios
    # plt.figure(figsize=(8, 6))
    plt.plot(epsilon_hist, color="red")
    plt.title("Decaimento do epsilon")
    plt.xlabel("Episódios")
    plt.ylabel("epsilon")
    plt.grid(True)
    plt.show()

    # Plotar a evolução da média geral da Q-Table
    # plt.figure(figsize=(8, 6))
    xi = range(len(q_table_history))
    plt.plot(xi, q_table_history, color="blue")
    plt.title("Evolução da Média Geral da Q-Table")
    plt.xlabel("Episódios")
    plt.ylabel("Média dos Valores Q")
    plt.grid(True)
    plt.show()

    # Gráfico Total rewards per episode
    plt.plot(rewards_per_episode)
    plt.xlabel('Rewards')
    plt.ylabel('Total rewards')
    plt.title('Total rewards per episode')
    plt.show()

    # rotina p/ cálculo da média móvel
    def media_movel(dados, janela):
      media_movel_lista = []
      for i in range(len(dados) - janela + 1):
        media_movel = sum(dados[i:i+janela])/janela
        media_movel_lista.append(media_movel)
      return media_movel_lista

    media_movel_ = media_movel(rewards_per_episode, janela=100)
    plt.plot(media_movel_, color="black", alpha=0.9)
    plt.title("Valor médio dos rewards")
    plt.xlabel("Episódios")
    plt.ylabel("média")
    plt.grid(True)
    plt.show()
