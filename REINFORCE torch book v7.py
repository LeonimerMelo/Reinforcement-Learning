# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:29:32 2025

@author: Leonimer
https://medium.com/@whyamit404/a-practical-guide-to-sampling-from-a-categorical-distribution-in-pytorch-a03a638c9cdb
"""
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame

# Pi constructs the policy network that is a a simple one-layer MLP with 64 hidden units
class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train() # set training mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    # act defines the method to produce action 
    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32)) # to tensor
        # x = torch.tensor(state, dtype=torch.float32) # same command
        pdparam = self.forward(x) # forward pass
        pd = Categorical(logits=pdparam) # probability distribution
        action = pd.sample() # pi(a|s) in action via pd
        # log_prob returns the logarithm of the density or probability
        log_prob = pd.log_prob(action) # log_prob of pi(a|s)
        self.log_probs.append(log_prob) # store for training
        return action.item()

'''
train implements the update steps in REINFORCE Algorithm Note that the loss is expressed
as the sum of the negative log probabilities multiplied by the returns. The negative sign 
is necessary because by default, PyTorch’s optimizer minimizes the loss, whereas we want 
to maximize the objective. Furthermore, we formulate the loss in this way to utilize 
PyTorch’s automatic differentiation feature. When we call loss.backward(), this computes
the gradient of the loss, which is equal to the policy gradient. Finally, we update the 
policy parameters by calling optimizer.step()
'''
def train(pi, optimizer, gamma):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32) # the returns
    G = 0.0 # future return
    # compute the returns efficiently
    for t in reversed(range(T)):
        G = pi.rewards[t] + gamma * G
        rets[t] = G
    rets = torch.tensor(rets)

    # Normaliza os retornos para melhorar a estabilidade numérica
    # 𝑧 = (𝑥−𝜇)/𝜎​
    # 𝑧 is the standardized value.
    # 𝑥 is the original value.
    # 𝜇 is the mean of the feature.
    # 𝜎 is the standard deviation of the feature.
    rets_std = (rets - rets.mean()) / (rets.std() + 1e-9)
    
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets_std # gradient term; Negative for maximizing
    # loss = - log_probs * rets # aqui temo o loss sem a normalização!
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward() # backpropagate, compute gradients
    optimizer.step() # gradient-ascent, update the weights
    return loss

# Cria o ambiente CartPole-v1
max_episode_steps_ = 3000
env = gym.make("CartPole-v1", render_mode=None, max_episode_steps = max_episode_steps_)

in_dim = env.observation_space.shape[0] # 4
out_dim = env.action_space.n # 2
pi = Pi(in_dim, out_dim) # policy pi_theta for REINFORCE
print(pi) # print model pi
print(list(pi.parameters())[0]) # print weights of the first hidden layer
print(list(pi.parameters())[0].shape)

optimizer = optim.Adam(pi.parameters(), lr=0.01)
print(optimizer) # print optimizer parameters

cont = 0
cont_solved = 0
solved = False
episodes = 600
gamma = 0.99
loss_hist = []
rewards_hist = []
episodes_run = 0
'''
The main loop. It constructs a CartPole environment, the policy network Pi,
and an optimizer. Then, it runs the training loop for 500 episodes (defaul). As training
progresses, the total reward per episode should increase towards max_episode_steps,
500 by default. The environment is solved when the total reward is araund max_episode_steps.
'''
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    truncated = False
    i = 0
    while not done and not truncated: # cartpole max timestep is 500
        action = pi.act(state)
        state, reward, done, truncated, _ = env.step(action)
        pi.rewards.append(reward)
        
        i += 1
        if solved:
            print(f'episode {episode + 1}, rewards: {i}', end='\r')
        
    episodes_run += 1
    loss = train(pi, optimizer, gamma) # train per episode
    loss_hist.append(loss.item())
    total_reward = sum(pi.rewards)
    rewards_hist.append(total_reward)
    pi.onpolicy_reset() # onpolicy: clear (self.log_probs and self.rewards) after training
    print("\033[K", end="\r") # clear current print line
    if not solved:
        print(f'episode {episode + 1}, loss: {loss:.2f}, total_reward: {total_reward}', end='\r')
    
    # if not solved: a partir de (episodes - 5) muda para render_mode='human'
    if episode == (episodes - 5):
        env.close()
        env = gym.make('CartPole-v1', render_mode='human')
        env.reset()
        
    # if solved: Para o treinamento se o agente resolver o ambiente (alcançar reward_threshold)
    if total_reward >= env.spec.max_episode_steps: 
        print(f"\nSolved in {episode + 1} episodes!")
        cont_solved += 1
        
    if cont_solved >= 5:    
        solved = True 
        
    if solved :
        cont += 1    
        
    if cont == 1:
        env.close()
        env = gym.make('CartPole-v1', render_mode='human', max_episode_steps = max_episode_steps_)
        env.reset()  
        
    if cont > 3:
        break

# encerra o ambiente de treinamento
env.reset()
env.close() 
pygame.quit()
    
mean_loss_ = []
for t in range(episodes_run):
    # calculo a média móvel das perdas
    mean_loss_.append(np.mean(loss_hist[max(0, t-episodes_run//3):(t+1)]))
# Plotar o histórico da função perdas (loss)
plt.title("Média Geral da função perda (loss)")
plt.xlabel("Episódios")
plt.ylabel("loss")
plt.grid(True)
plt.plot(mean_loss_, color="red")
plt.show()

mean_rewards_ = []
for t in range(episodes_run):
    # calculo a média móvel dos rewards
    mean_rewards_.append(np.mean(rewards_hist[max(0, t-episodes_run//7):(t+1)]))
# Plotar o histórico das recompensas
plt.title("Evolução das recompensas (rewards) por episódio")
plt.xlabel("Episódios")
plt.ylabel("rewards")
plt.grid(True)
plt.plot(mean_rewards_, color="black")
plt.show()
