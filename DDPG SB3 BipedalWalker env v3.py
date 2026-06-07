# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:30:55 2026

@author: Leonimer

https://gymnasium.farama.org/environments/box2d/bipedal_walker/

Bipedal Walker - Description
============================
This is a simple 4-joint walker robot environment. There are two versions:
Normal, with slightly uneven terrain.
Hardcore, with ladders, stumps, pitfalls.
To solve the normal version, you need to get 300 points in 1600 time steps. 
To solve the hardcore version, you need 300 points in 2000 time steps.

Action Space
===========
Actions are motor speed values in the [-1, 1] range for each of the 4 joints at 
both hips and knees.

Observation Space
=================
State consists of hull angle speed, angular velocity, horizontal speed, vertical 
speed, position of joints and joints angular speed, legs contact with ground, 
and 10 lidar rangefinder measurements. There are no coordinates in the state vector.

Rewards
=======
Reward is given for moving forward, totaling 300+ points up to the far end. 
If the robot falls, it gets -100. Applying motor torque costs a small amount 
of points. A more optimal agent will get a better score.
"""

'''
https://spinningup.openai.com/en/latest/algorithms/ddpg.html

O DDPG (Deep Deterministic Policy Gradient)
===========================================
O DDPG é um algoritmo que aprende simultaneamente uma função Q e uma política. 
Ele utiliza dados fora da política e a equação de Bellman para aprender a função Q 
e, em seguida, utiliza a função Q para aprender a política.

É um algoritmo Reinforcement Learning voltado para espaços de ação contínuos, 
como controle de motores robóticos ou condução autônoma, onde as ações não são 
discretas (como "virar para a esquerda"), mas valores precisos (como "virar 
π/4 radianos"). Ele combina o sucesso das redes neurais profundas com a 
abordagem Actor-Critic e gradientes de política determinísticos.

O DDPG opera através de duas redes neurais principais que trabalham juntas:
O Ator (Actor): Mapeia o estado atual diretamente para uma ação determinística 
(a melhor ação para aquele momento).
O Crítico (Critic): Avalia a qualidade da ação escolhida pelo Ator, gerando um 
valor Q (expectativa de recompensa futura) para um determinado par de estado-ação.

Principais Componentes do Algoritmo
===================================
Aprendizado fora da política (Off-Policy): Ele não aprende apenas com as experiências 
atuais, mas também com interações passadas armazenadas em um banco de dados chamado 
Replay Buffer. Isso melhora muito a eficiência amostral.
Redes Alvo (Target Networks): O DDPG utiliza cópias "lentas" dos parâmetros do 
Ator e do Crítico. Isso estabiliza o processo de aprendizado, evitando que a mudança 
abrupta em uma rede desestabilize a outra
'''

import gymnasium as gym
import cv2
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy

train = True
train = False

time_steps = int(100_000)
path='C:\\Leo\\python scripts\\ddpg_BipedalWalker_'+str(time_steps)

# Create environment
env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array") 
               # max_episode_steps=2000)

if train:
    # Instantiate the agent
    model = DDPG("MlpPolicy", env, verbose=1, learning_rate=1e-4)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=time_steps, progress_bar=True)
    # Save the agent
    model.save(path)
    
    del model  # delete trained model to demonstrate loading

if not train:
    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    model = DDPG.load(path, env=env, print_system_info=True)
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
    # for i in range(10_000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = vec_env.step(action)
    #     rewards_ += rewards.item()
    #     vec_env.render("human")
    #     if dones:
    #         print('rewards:', int(rewards_), 'episode:', episode)
    #         rewards_ = 0
    #         episode += 1
            

    while episode < 6:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        rewards_ += rewards.item()
        vec_env.render("human")
        if dones:
            print('episode:', episode, 'rewards:', int(rewards_))
            rewards_ = 0
            episode += 1
            
    vec_env.close()

env.close()
cv2.destroyAllWindows()