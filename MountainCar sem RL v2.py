# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 12:02:56 2024

@author: Leonimer

MountainCar sem Reinforcement Learning!
"""

import gymnasium as gym

# Cria o ambiente
env = gym.make('MountainCar-v0', render_mode='human')

# Reinicia o ambiente
env.reset()
action = 1
for i in range(500):  # time steps
    if i < 2:   # empurrãozinho inicial!
        action = 2
    next_state, reward, done, truncated, info = env.step(action)  # Executa a ação no ambiente
    position = next_state[0]
    velocity = next_state[1]
    
    if velocity > 0 and position > -0.5: # empurrãozinho para direita!
        action = 2
    elif velocity < 0 and position < -0.5:  # empurrãozinho para esquerda!
        action = 0
    else:
        action = 1
        
    print('time steps:', i, end='\r')
        
    if done:
        print("\nObjetivo alcançado!")
        break

env.close()
