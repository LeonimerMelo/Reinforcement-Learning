# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:37:14 2025

@author: TechnoLEDs
https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
"""

from pathlib import Path
import pybullet_envs_gymnasium
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import pygame

# Alternatively, you can use the MuJoCo equivalent "HalfCheetah-v4"
vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
# Automatically normalize the input features and reward
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu')
model.learn(total_timesteps=500_000)

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = Path("/tmp/")

model.save(log_dir / "ppo_halfcheetah")
stats_path = log_dir / "vec_normalize.pkl"
vec_env.save(stats_path)

# To demonstrate loading
del model, vec_env

# Load the saved statistics
vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
vec_env = VecNormalize.load(stats_path, vec_env)
#  do not update them at test time
vec_env.training = False
# reward normalization is not needed at test time
vec_env.norm_reward = False

# Load the agent
model = PPO.load(log_dir / "ppo_halfcheetah", env=vec_env, device='cpu')


obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

mean_reward, std_reward = evaluate_policy(model, vec_env)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

vec_env.close()
pygame.quit()

