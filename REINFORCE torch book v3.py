# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:29:32 2025

@author: TechnoLEDs
"""
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


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
        pdparam = self.forward(x) # forward pass
        pd = Categorical(logits=pdparam) # probability distribution
        action = pd.sample() # pi(a|s) in action via pd
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
def train(pi, optimizer):
    gamma = 0.99
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32) # the returns
    future_ret = 0.0
    # compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets # gradient term; Negative for maximizing
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward() # backpropagate, compute gradients
    optimizer.step() # gradient-ascent, update the weights
    return loss

'''
The main loop. It constructs a CartPole environment, the policy network Pi,
and an optimizer. Then, it runs the training loop for 500 episodes. As training
progresses, the total reward per episode should increase towards 500. The
environment is solved when the total reward is above 495.
'''
env = gym.make('CartPole-v1', render_mode=None)
in_dim = env.observation_space.shape[0] # 4
out_dim = env.action_space.n # 2
pi = Pi(in_dim, out_dim) # policy pi_theta for REINFORCE
optimizer = optim.Adam(pi.parameters(), lr=0.01)
episodes = 1500
loss_hist = []
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated: # cartpole max timestep is 500
        action = pi.act(state)
        state, reward, done, truncated, _ = env.step(action)
        pi.rewards.append(reward)
        
    loss = train(pi, optimizer) # train per episode
    loss_hist.append(loss.item())
    total_reward = sum(pi.rewards)
    solved = total_reward > 495.0
    pi.onpolicy_reset() # onpolicy: clear memory after training
    print("\033[K", end="\r") # clear current print line
    print(f'Episode {episode}, loss: {loss:.2f}, total_reward: {total_reward}, solved: {solved}', end='\r')

    if episode == (episodes - 3):
        env.close()
        env = gym.make('CartPole-v1', render_mode='human')
        env.reset()

env.close()
    

# Plotar o histórico da função perdas (loss)
plt.figure(figsize=(8, 6))
# plt.plot(loss_hist, color="blue")
plt.title("Média Geral da função perda (loss)")
plt.xlabel("Episódios")
plt.ylabel("loss")
#plt.legend()
plt.grid(True)
# plt.show()

mean_loss_ = []
for t in range(episodes):
    # calculo a média móvel dos rewards de 30 episódios
    mean_loss_.append(np.mean(loss_hist[max(0, t-30):(t+1)]))
# plt.title('Mean rewards per episode')
# plt.xlabel('episodes')
# plt.ylabel('rewards')
plt.plot(mean_loss_, color="red")
plt.show()
    
# if __name__ == '__main__':
#     main()



# env = gym.make('CartPole-v1', render_mode='human')
# env = gym.make('CartPole-v1', render_mode=None)
# env.reset()
# env.step(1)
# env.close()
