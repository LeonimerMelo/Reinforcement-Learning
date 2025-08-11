# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 14:21:56 2025

@author: Leonimer
https://github.com/bentrevett/pytorch-dqn/blob/master/1_dqn.ipynb
https://github.com/Farama-Foundation/Arcade-Learning-Environment/tree/master
"""

import collections
import copy
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ale_py
import pygame

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
np.random.seed(1234)
random.seed(1234)

# game = 'PongNoFrameskip-v4'
game='ALE/Pong-v5'

train_ = False
render='human'
if train_:
    render='rgb_array'
    print('Trainning agent')
else:
    print('Testing agent')
    
env1 = gym.make(game, frameskip=1, render_mode=render)
obs, info=env1.reset()
print(obs.shape)
plt.imshow(obs)
plt.show()

env2 = gym.wrappers.AtariPreprocessing(env1,
                                      noop_max=30,
                                      frame_skip=4,
                                      screen_size=84,
                                      terminal_on_life_loss=True,
                                      grayscale_obs=True)

obs, info=env2.reset()
print(obs.shape)
plt.imshow(obs)
plt.show()

n_stack = 4
env3 = gym.wrappers.FrameStackObservation(env=env2, stack_size=n_stack)

obs, info=env3.reset()
print(obs.shape)

fig, ax = plt.subplots(1, 4, figsize=(15,15))
for i, s in enumerate(obs):
    ax[i].imshow(s)
plt.show()   
    
env4 = gym.wrappers.TransformReward(env3, lambda r: np.sign(r))
print(env4.action_space, env4.action_space.n)

action = 0
for i in range(25):
    next_state, reward, done, truncate, info = env4.step(action)
fig, ax = plt.subplots(1, 4, figsize=(15,15))
for i, s in enumerate(next_state):
    ax[i].imshow(s)
plt.show()
    
class DQN(nn.Module):
    def __init__(self, n_stack: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(n_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        batch_size, n_stack, height, width = x.shape
        assert (height, width) == (84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_action(self, state, epsilon, device):
        if random.random() < epsilon:
            action = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(np.array(state)).unsqueeze(dim=0).to(device)
                q_value = self.forward(state)
                action = q_value.argmax(dim=-1).item()
        return action

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor([done])
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)
        
    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.memory)
    
def optimize_model(model, target_model, replay_memory, optimizer, batch_size, gamma, grad_clip, device):
    
    states, actions, rewards, next_states, dones = replay_memory.sample(batch_size)
    
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    rewards = torch.stack(rewards).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.stack(dones).to(device)
        
    q_preds = model(states)
    q_vals = q_preds.gather(dim=-1, index=actions)
    
    target_preds = target_model(next_states)
    target_vals = target_preds.max(dim=-1, keepdim=True).values
    
    expected_vals = rewards + (target_vals * gamma * (1 - dones))
    
    loss = F.smooth_l1_loss(q_vals, expected_vals.detach())
            
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    
    return loss.item()
    
def train(env, model, replay_memory, optimizer, n_steps, update_freq,
          target_freq, print_freq, epsilons, batch_size, gamma, grad_clip,
          device):
    
    model = model.to(device)
    target_model = copy.deepcopy(model)
    model.train()
    model.eval()
    
    episode_steps = []
    episode_rewards = []
    episode_reward = 0

    state, info = env.reset()

    for step in tqdm.tqdm(range(n_steps)):

        epsilon = epsilons[step] if step < len(epsilons) else epsilons[-1]

        action = model.get_action(state, epsilon, device)

        next_state, reward, done, truncate, info = env.step(action)

        episode_reward += reward

        replay_memory.push(state, action, reward, next_state, done or truncate)

        state = next_state

        if step % update_freq == 0 and step > replay_memory.capacity:
            loss = optimize_model(model, target_model, replay_memory, optimizer,
                                  batch_size, gamma, grad_clip, device)

        if step % (update_freq * target_freq) == 0 and step > replay_memory.capacity:
            target_model.load_state_dict(model.state_dict())

        if done or truncate:
            episode_rewards.append(episode_reward)
            episode_steps.append(step+1)
            episode_reward = 0
            state, info = env.reset()
            
        if step % print_freq == 0 and step > replay_memory.capacity:
            avg_reward = np.mean(episode_rewards[-10:])
            print(' ')
            print(f'episodes: {len(episode_rewards)}, steps: {step}')
            print(f'epsilon: {epsilon}, reward: {avg_reward}')
            
    return episode_steps, episode_rewards

capacity = 10_000
replay_memory = ReplayMemory(capacity)

start = 1.0
end = 0.01
decay = 30_000
epsilons = np.concatenate((np.ones(capacity), np.linspace(start, end, decay)), axis=0)
plt.plot(epsilons)
plt.show()

n_actions = env4.action_space.n
model = DQN(n_stack, n_actions)

lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# n_steps = 1_000_000
n_steps = 300_000
update_freq = 1
target_freq = 1_000
print_freq = 1_000
gamma = 0.99
batch_size = 32
grad_clip = 1.0
test_steps = 10_000
test_epsides = 2
path='C:\\Leo\\python scripts\\'

if train_:
    train(env4, model, replay_memory, optimizer, n_steps, update_freq,
          target_freq, print_freq, epsilons, batch_size, gamma, grad_clip, device)

    torch.save(model.state_dict(), path + 'model_DQN_Pong_'+str(n_steps)+'.pth')
    print("Saved PyTorch Model State to 'model_DQN_Pong.pth'")

else:
    '''
    Loading Model for inference
    ===========================
    '''
    model = DQN(n_stack, n_actions).to(device)
    model.load_state_dict(torch.load(path + 'model_DQN_Pong_'+str(n_steps)+'.pth', weights_only=True))
    print("Loaded PyTorch Model State from 'model_DQN_Pong.pth'")
    
    rewards=0
    epsilon = 0.01
    for episode in range(test_epsides):
        state, info = env4.reset()
        for step in range(test_steps):
            action = model.get_action(state, epsilon, device)
            next_state, reward, done, truncate, info = env4.step(action)
            state = next_state
            rewards += reward
            if done or truncate:
                print('Episode:', episode+1, 'Rewords:', rewards)
                break
                
    env4.close() 
    pygame.quit()
