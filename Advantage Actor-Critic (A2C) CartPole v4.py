# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:21:11 2025

@author: TechnoLEDs
"""

# implementação do Advantage Actor-Critic (A2C) em PyTorch
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Configurações
# ENV_NAME = "CartPole-v1"
# env = gym.make(ENV_NAME)
max_episode_steps_ = 1000
env = gym.make('CartPole-v1', render_mode = 'rgb_array',
               max_episode_steps = max_episode_steps_)
GAMMA = 0.99
LR = 1e-3
N_STEPS = 10
EPISODES = 1400

# Rede Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        # Cabeça da Política (Ator)
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Cabeça do Crítico (Valor)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        shared = self.shared_layers(state)
        policy = self.actor(shared)
        value = self.critic(shared)
        return policy, value

# Agente A2C
class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.model = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, _ = self.model(state)
        action = torch.multinomial(policy, num_samples=1).item()
        return action
    
    def compute_advantage(self, rewards, values, next_value, dones):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + GAMMA * R * (1 - dones[step])
            returns.insert(0, R)
        returns = torch.tensor(returns)
        advantage = returns - values
        return returns, advantage

    def train(self, trajectory):
        states, actions, rewards, dones, next_state = trajectory
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        # Predições do modelo
        _, next_value = self.model(next_state)
        policy, values = self.model(states)

        # Obtem a função Advantage
        values = values.squeeze()
        returns, advantage = self.compute_advantage(rewards, values, next_value.item(), dones)

        # Calcula perdas
        log_probs = torch.log(policy.gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        # Otimização
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Loop de Treinamento
rewards_hist = []
steps_hist = []
def train_a2c():
    # env = gym.make(ENV_NAME)
    agent = A2CAgent(env)
    for episode in range(EPISODES):
        state, _ = env.reset()  # Gymnasium retorna estado e info
        trajectory = {"states": [], "actions": [], "rewards": [], "dones": []}
        episode_reward = 0

        for t in range(1, 10000):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Certifique-se de que o estado é uma lista ou ndarray
            state = np.array(state, dtype=np.float32).tolist()
            next_state = np.array(next_state, dtype=np.float32).tolist()

            # Armazena a experiência
            trajectory["states"].append(state)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done or truncated)
            state = next_state
            episode_reward += reward

            # Atualiza o agente a cada N passos
            if len(trajectory["rewards"]) == N_STEPS or done or truncated:
                trajectory["next_state"] = state
                agent.train((
                    trajectory["states"], 
                    trajectory["actions"], 
                    trajectory["rewards"], 
                    trajectory["dones"], 
                    trajectory["next_state"]
                ))
                trajectory = {"states": [], "actions": [], "rewards": [], "dones": []}
                
            total_reward = int(sum(trajectory["rewards"]))
            rewards_hist.append(total_reward)

            if done or truncated:
                print("\033[K", end="\r") # clear current print line
                print(f"Episode {episode}, Reward: {episode_reward}", end="\r")
                break
            
        steps_hist.append(t)
      
# Treinando o agente
# if __name__ == "__main__":
train_a2c()
env.close()

# rotina p/ cálculo da média móvel
def media_movel(dados, janela):
  media_movel_lista = []
  for i in range(len(dados) - janela + 1):
    media_movel = sum(dados[i:i+janela])/janela
    media_movel_lista.append(media_movel)
  return media_movel_lista

janela_=100
media_movel_ = media_movel(steps_hist, janela=janela_)
media_movel__ = [0]*janela_ + media_movel_
plt.plot(media_movel__, color="red", alpha=0.9, lw=2)
plt.plot(steps_hist, alpha=0.7)
plt.title("Valor médio dos time steps")
plt.xlabel("Episódios")
plt.ylabel("média")
plt.grid(True)
plt.show()

mean_rewards_ = []
for t in range(EPISODES):
    # calculo a média móvel dos rewards
    mean_rewards_.append(np.mean(rewards_hist[max(0, t-30):(t+1)]))
# Plotar o histórico das recompensas
plt.title("Evolução das recompensas (rewards) por episódio")
plt.xlabel("Episódios")
plt.ylabel("rewards")
plt.grid(True)
plt.plot(mean_rewards_, color="black", alpha=.7)
plt.show()

# avaliação
EPISODES = 30
env = gym.make('CartPole-v1', render_mode='human')
env.reset()
train_a2c()
env.close()
