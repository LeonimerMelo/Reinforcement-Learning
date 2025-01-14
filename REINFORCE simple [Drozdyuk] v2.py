import torch
import gymnasium as gym
import pygame
import numpy as np
import matplotlib.pyplot as plt

max_episode_steps_ = 2000
env = gym.make("CartPole-v1", render_mode=None, max_episode_steps = max_episode_steps_)

nn = torch.nn.Sequential(
    torch.nn.Linear(4, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, env.action_space.n),
    torch.nn.Softmax(dim=-1)
)
learnig_rate = 0.005
optim = torch.optim.Adam(nn.parameters(), lr=learnig_rate)

episodes = 300
gamma = 0.9999
cont = 0
solved = False
rewards_hist = []
for episode in range(episodes):  # Loop sobre os episódios
    obs = torch.tensor(env.reset()[0], dtype=torch.float)    
    done = False
    Actions, States, Rewards = [], [], []
    steps = 0
    while not done:
        probs = nn(obs)
        dist = torch.distributions.Categorical(probs=probs)        
        action = dist.sample().item()
        obs_, rew, terminated, truncated, _ = env.step(action)
        
        Actions.append(torch.tensor(action, dtype=torch.int))
        States.append(obs)
        Rewards.append(rew)

        obs = torch.tensor(obs_, dtype=torch.float)
        steps += 1
        done = terminated or truncated
        
    DiscountedReturns = []
    for t in range(len(Rewards)):
        G = 0.0
        for k, r in enumerate(Rewards[t:]):
            G += (gamma**k) * r
        DiscountedReturns.append(G)
    
    for State, Action, G in zip(States, Actions, DiscountedReturns):
        probs = nn(State)
        dist = torch.distributions.Categorical(probs=probs)    
        log_prob = dist.log_prob(Action)
        
        loss = - log_prob * G
        
        optim.zero_grad()
        loss.backward()
        optim.step()
    
    total_reward = int(sum(Rewards))
    rewards_hist.append(total_reward)
    print("\033[K", end="\r") # clear current print line
    print(f"Episode {episode + 1}/{episodes}  \tLoss: {loss.item():.2f}  \tTotal Reward: {total_reward}", end='\r')

    # Para o treinamento se o agente resolver o ambiente (alcançar reward_threshold)
    if total_reward >= env.spec.max_episode_steps:  
        print(f"\nSolved in {episode + 1} episodes!")
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

mean_rewards_ = []
for t in range(episodes-1):
    # calculo a média móvel dos rewards
    mean_rewards_.append(np.mean(rewards_hist[max(0, t-episodes//6):(t+1)]))
# Plotar o histórico das recompensas
plt.title("Evolução das recompensas (rewards) por episódio")
plt.xlabel("Episódios")
plt.ylabel("rewards")
plt.grid(True)
plt.plot(mean_rewards_, color="black")
plt.show()

