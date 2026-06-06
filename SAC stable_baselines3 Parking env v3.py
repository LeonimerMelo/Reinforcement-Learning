# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:42:33 2025

@author: Leonimer
https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
https://github.com/ghoshavirup0/highwayEnv
https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/parking_model_based.ipynb
"""

import gymnasium as gym
import highway_env
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make("parking-v0")

# Create 4 artificial transitions per real transition
n_sampled_goal = 4

# SAC hyperparams:
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
      n_sampled_goal=n_sampled_goal,
      goal_selection_strategy="future",
    ),
    verbose=1,
    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=256,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
)

model.learn(int(50_000))
path='C:\\Leo\\python scripts\\her_sac_parking_2'
model.save(path)

# Load saved model
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
env = gym.make("parking-v0", render_mode="human") # Change the render mode
model = SAC.load(path, env=env)

obs, info = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    if terminated or truncated or info.get("is_success", False):
        print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        episode_reward = 0.0
        obs, info = env.reset()
        
env.close()
