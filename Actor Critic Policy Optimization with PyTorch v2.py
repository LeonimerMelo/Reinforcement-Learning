# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:52:48 2025

@author: Leonimer
https://github.com/Apress/deep-reinforcement-learning-python/blob/main/chapter7/listing7_2_actor_critic_pytorch.ipynb

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
# from scipy import signal

max_episode_steps_ = 2000
env = gym.make('CartPole-v1', render_mode = 'rgb_array',
               max_episode_steps = max_episode_steps_)

env.reset()
plt.imshow(env.render())
plt.show()

state_shape, n_actions = env.observation_space.shape, env.action_space.n
state_dim = state_shape[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128,n_actions)
        self.critic = nn.Linear(128,1)


    def forward(self, s):
        x = F.relu(self.fc1(s))
        logits = self.actor(x)
        state_value = self.critic(x)
        return logits, state_value
        
model = ActorCritic()
model = model.to(device)

def sample_action(state):
    """
    params: states: [batch, state_dim]
    returns: probs: [batch, n_actions]
    """
    state = torch.tensor(state, device=device, dtype=torch.float32)
    with torch.no_grad():
        logits,_ = model(state)
    # action_probs = nn.functional.softmax(logits, -1).detach().numpy()[0]
    action_probs = nn.functional.softmax(logits, -1).cpu().numpy()[0]
    action = np.random.choice(n_actions, p=action_probs)
    return action

def generate_trajectory(env, n_steps=2000):
    """
    Play a session and genrate a trajectory
    returns: arrays of states, actions, rewards
    """
    states, actions, rewards = [], [], []
    
    # initialize the environment
    s,_ = env.reset()
    
    #generate n_steps of trajectory:
    for steps in range(n_steps):
        #sample action based on action_probs
        a = sample_action(np.array([s]))
        next_state, r, done, truncated, _ = env.step(a)
        
        #update arrays
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = next_state
        if done or truncated:
            break
    
    return states, actions, rewards, steps

def get_rewards_to_go(rewards, gamma=0.99):
    
    T = len(rewards) # total number of individual rewards
    # empty array to return the rewards to go
    rewards_to_go = [0]*T 
    rewards_to_go[T-1] = rewards[T-1]
    
    for i in range(T-2, -1, -1): #go from T-2 to 0
        rewards_to_go[i] = gamma * rewards_to_go[i+1] + rewards[i]
    
    return rewards_to_go

#init Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_episode(states, actions, rewards, gamma=0.99, entropy_coef=1e-2):
    
    # get rewards to go
    rewards_to_go = get_rewards_to_go(rewards, gamma)

    states = np.array(states)
    actions = np.array(actions)
    rewards_to_go = np.array(rewards_to_go)
    
    # convert numpy array to torch tensors
    states = torch.tensor(states, device=device, dtype=torch.float)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards_to_go = torch.tensor(rewards_to_go, device=device, dtype=torch.float)
    # states = torch.tensor(np.array(states), device=device, dtype=torch.float)
    # actions = torch.tensor(np.array(actions), device=device, dtype=torch.long)
    # rewards_to_go = torch.tensor(np.array(rewards_to_go), device=device, dtype=torch.float)
    
    # get action probabilities from states
    logits, state_values = model(states)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)
    
    log_probs_for_actions = log_probs[range(len(actions)), actions]
    advantage = rewards_to_go - state_values.squeeze(-1)
    
    #Compute loss to be minized
    J = torch.mean(log_probs_for_actions*(advantage))
    H = -(probs*log_probs).sum(-1).mean()
    
    loss = -(J+entropy_coef*H)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return np.sum(rewards) #to show progress on training

print('\nTreinando o agente')
EPSODES = 1000
total_rewards = []
for i in range(EPSODES):
    states, actions, rewards, steps = generate_trajectory(env)
    reward = train_one_episode(states, actions, rewards)
    total_rewards.append(reward)
    if i != 0 and i % 50 == 0:
        mean_reward = np.mean(total_rewards[-50:-1])
        print("epsode: %d  mean reward: %.1f" % (i, mean_reward))
        if mean_reward > 850:
            break

env.close()

# rotina p/ cálculo da média móvel
def media_movel(dados, janela):
  media_movel_lista = []
  for i in range(len(dados) - janela + 1):
    media_movel = sum(dados[i:i+janela])/janela
    media_movel_lista.append(media_movel)
  return media_movel_lista

janela_=100
media_movel_ = media_movel(total_rewards, janela=janela_)
media_movel__ = [0]*janela_ + media_movel_
plt.plot(media_movel__, color="red", alpha=0.9, lw=2)
plt.plot(total_rewards, alpha=0.7)
plt.title("Valor médio dos rewards")
plt.xlabel("Episódios")
plt.ylabel("Rewards")
plt.grid(True)
plt.show()

print('\nAvaliando o agente')
env = gym.make('CartPole-v1', render_mode='human', max_episode_steps = max_episode_steps_)
env.reset()
for i in range(10):
    states, actions, rewards, steps = generate_trajectory(env)
    print('epsode:', i+1, 'steps:', steps+1)

env.close()