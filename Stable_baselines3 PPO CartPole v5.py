# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 10:29:26 2026

@author: TechnoLEDs
"""

# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv

env_ = "Acrobot-v1"
# Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=4, vec_env_cls=SubprocVecEnv)
# vec_env = make_vec_env("CartPole-v1", n_envs=4)
vec_env = make_vec_env(env_, n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu')
model.learn(total_timesteps=500_000)
model.save("ppo_"+env_+str(vec_env.num_envs))

del model # remove to demonstrate saving and loading

path = "C:\Leo\python scripts\\"
model = PPO.load(path+"ppo_"+env_+str(vec_env.num_envs), device='cpu')

obs = vec_env.reset()
epsodes = 0
rewards_ = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    rewards_ += 1
    
    if dones[0]:
        epsodes += 1
        print('epsodes:', epsodes, 'rewards:', rewards_)
        rewards_ = 0
        
    if epsodes > 2:
        break
    
vec_env.close()

# Force each individual environment to execute its close logic
# vec_env.env_method("close")
# vec_env.close()

# import pygame
# pygame.display.quit()
# pygame.quit()
