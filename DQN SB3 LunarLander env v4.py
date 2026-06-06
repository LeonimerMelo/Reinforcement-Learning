# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:30:55 2026

@author: Leonimer

https://gymnasium.farama.org/environments/box2d/lunar_lander/

Description
===========
This environment is a classic rocket trajectory optimization problem. 
According to Pontryagin’s maximum principle, it is optimal to fire the engine 
at full throttle or turn it off. This is the reason why this environment has 
discrete actions: engine on or off.

There are two environment versions: discrete or continuous. The landing pad 
is always at coordinates (0,0). The coordinates are the first two numbers 
in the state vector. Landing outside of the landing pad is possible. Fuel is 
infinite, so an agent can learn to fly and then land on its first attempt.

Action Space
============
There are four discrete actions available:
0: do nothing
1: fire left orientation engine
2: fire main engine
3: fire right orientation engine

Observation Space
=================
The state is an 8-dimensional vector: the coordinates of the lander in x & y, 
its linear velocities in x & y, its angle, its angular velocity, and two booleans 
that represent whether each leg is in contact with the ground or not.

Rewards
=======
After every step a reward is granted. The total reward of an episode is the 
sum of the rewards for all the steps within that episode.
For each step, the reward:
is increased/decreased the closer/further the lander is to the landing pad.
is increased/decreased the slower/faster the lander is moving.
is decreased the more the lander is tilted (angle not horizontal).
is increased by 10 points for each leg that is in contact with the ground.
is decreased by 0.03 points each frame a side engine is firing.
is decreased by 0.3 points each frame the main engine is firing.
The episode receive an additional reward of -100 or +100 points for crashing or 
landing safely respectively.

An episode is considered a solution if it scores at least 200 points.
"""

import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

train = True
train = False

time_steps = int(500_000)
path='C:\\Leo\\python scripts\\dqn_lunar_'+str(time_steps)

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

if train:
    # Instantiate the agent
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-4)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=time_steps, progress_bar=True)
    # Save the agent
    model.save(path)
    
    del model  # delete trained model to demonstrate loading

if not train:
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    model = DQN.load(path, env=env, print_system_info=True)
    # model = DQN.load("dqn_lunar_50_000", env=env)
    
    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap the environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    print('mean reward:', mean_reward, 'std reward:', std_reward)
    
    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    rewards_ = 0
    episode = 1
    for i in range(10_000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        rewards_ += rewards.item()
        vec_env.render("human")
        if dones:
            print('rewards:', int(rewards_), 'episode:', episode)
            rewards_ = 0
            episode += 1
            
    vec_env.close()

env.close()
