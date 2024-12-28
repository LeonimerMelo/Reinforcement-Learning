# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:20:07 2024

@author: TechnoLEDs

https://github.com/nicknochnack/TensorflowKeras-ReinforcementLearning
"""

'''
CartPole
========
The main features of the environment are summarized below:
1. Objective: Keep the pole upright for 200 time steps.
2. State: An array of length 4 which represents: [cart position, cart velocity, pole angle,
pole angular velocity]. For example, [−0.034, 0.032, −0.031, 0.036].
3. Action: An integer, either 0 to move the cart a fixed distance to the left, or 1 to
move the cart a fixed distance to the right.
4. Reward: +1 for every time step the pole remains upright.
5. Termination: When the pole falls over (greater than 12 degrees from vertical), or
when the cart moves out of the screen, or when the maximum time step of 200 is
reached
'''

import gymnasium as gym
#import gym 
import random
import pygame

pygame.init()

env = gym.make('CartPole-v1', render_mode="human")
#env = gym.make('CartPole-v1', render_mode="rgb_array")
states = env.observation_space.shape[0]
actions = env.action_space.n

print(states)
print(actions)
print(env.spec)
print(env.metadata)
#print(gym.envs.registry)

for key in vars(env.spec):
    print('%s: %s' % (key, vars(env.spec)[key]))

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    timeStep=0
    
    while not done:
        env.render()
        #action = random.choice([0,1])
        action=random.randint(0,1)
        #_, n_state, reward, done, info = env.step(action)
        
        # Take a step in the environment
        step_result = env.step(action)

        # Check the number of values returned and unpack accordingly
        if len(step_result) == 4:
            next_state, reward, done, info = step_result
            terminated = False
        else:
            next_state, reward, done, truncated, info = step_result
            terminated = done or truncated

        score+=reward
        timeStep+=1
        
        if terminated:
            state = env.reset()  # Reset the environment if the episode is finished
            if truncated: 
                print('truncated')
        
        print('time step:', timeStep)
        print(f"Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}, Info: {info}")   
        
    print('Episode:{} Score:{}'.format(episode, score))

    
env.close()
