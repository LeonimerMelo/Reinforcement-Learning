'''
Atari Breakout

código para implementar um agente DQN (Deep Q-Network) completo para Atari Breakout.

Estrutura Geral do Código:
==========================
Rede Neural (DQN): Uma CNN que processa os frames do jogo e estima os valores Q 
para cada ação possível.
Buffer de Replay: Armazena as experiências passadas (estado, ação, recompensa, 
próximo estado) para treinamento.
Pré-processamento: Converte os frames do jogo para escala de cinza e redimensiona 
para 84x84 para facilitar o processamento.
Agente DQN: Implementa o algoritmo Q-learning com redes neurais, incluindo 
exploração epsilon-greedy.

Conceitos-Chave Implementados:
==============================
Redes duplas (policy e target): Estabiliza o treinamento ao separar a rede de 
predição da rede alvo.
Política epsilon-greedy: Equilibra exploração e aproveitamento durante o treinamento.
Experience replay: Permite aprendizado a partir de experiências passadas, quebrando 
correlações temporais.
Huber loss: Para robustez contra outliers durante o treinamento.
'''
# pip install gym[accept-rom-license]
# pip install gymnasium[atari]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pygame
import ale_py
import shimmy
import time

# Verificar se CUDA está disponível e definir o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Classe para a rede neural que será nossa Q-Network
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        """
        Inicializa a arquitetura da rede neural
        
        Args:
            input_shape: Forma da entrada (canais, altura, largura)
            n_actions: Número de ações possíveis
        """
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # Primeira camada convolucional
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Segunda camada convolucional
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Terceira camada convolucional
            nn.ReLU()
        )
        
        # Calculamos o tamanho do output após as camadas convolucionais
        conv_out_size = self._get_conv_output(input_shape)
        
        # Camadas totalmente conectadas (fully connected)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),  # Camada densa
            nn.ReLU(),
            nn.Linear(512, n_actions)  # Camada de saída com valores Q para cada ação
        )
    
    def _get_conv_output(self, shape):
        """
        Calcula a dimensão do output após as camadas convolucionais
        
        Args:
            shape: Forma da entrada
            
        Returns:
            int: Número de features após as camadas convolucionais
        """
        # Cria um tensor de exemplo para passar pelas camadas convolucionais
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """
        Forward pass através da rede neural
        
        Args:
            x: Entrada (estado do jogo)
            
        Returns:
            Valores Q para cada ação possível
        """
        # x tem formato (batch, canais, altura, largura)
        conv_out = self.conv(x).view(x.size()[0], -1)  # Achata o output da convolução
        return self.fc(conv_out)  # Passa pelo fully connected

# Classe para o buffer de experiência
class ReplayBuffer:
    def __init__(self, capacity):
        """
        Inicializa o buffer de experiência com a capacidade especificada
        
        Args:
            capacity: Tamanho máximo do buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Adiciona uma transição ao buffer
        
        Args:
            state: Estado atual
            action: Ação tomada
            reward: Recompensa recebida
            next_state: Próximo estado
            done: Flag indicando se o episódio terminou
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Amostra aleatoriamente um batch de transições do buffer
        
        Args:
            batch_size: Tamanho do batch
            
        Returns:
            Tuple contendo batches de estados, ações, recompensas, próximos estados e flags 'done'
        """
        # Seleciona 'batch_size' amostras aleatórias do buffer
        transitions = random.sample(self.buffer, batch_size)
        
        # Descompacta as transições
        batch = list(zip(*transitions))
        
        # Converte para tensores e retorna
        states = torch.cat(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.int64, device=device)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=device)
        next_states = torch.cat(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.bool, device=device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Retorna o tamanho atual do buffer
        """
        return len(self.buffer)

# Pré-processamento das observações do ambiente
class AtariPreprocessing:
    def __init__(self, env, skip_frames=4, grayscale=True):
        """
        Inicializa o wrapper para pré-processamento
        
        Args:
            env: Ambiente Gym
            skip_frames: Número de frames para pular (frame skipping)
            grayscale: Se deve converter para escala de cinza
        """
        self.env = env
        self.skip_frames = skip_frames
        self.grayscale = grayscale
        
        # Forma da observação processada
        self.observation_shape = (1, 84, 84) if grayscale else (3, 84, 84)
    
    def reset(self):
        """
        Reinicia o ambiente e retorna a observação inicial processada
        
        Returns:
            Observação inicial processada
        """
        observation, info = self.env.reset()
        return self.process_observation(observation), info
    
    def step(self, action):
        """
        Executa a ação, pulando frames, e retorna a observação processada
        
        Args:
            action: Ação a ser tomada
            
        Returns:
            Observação processada, recompensa acumulada, flag 'done' e info
        """
        total_reward = 0.0
        done = False
        info = {}
        
        # Repete a mesma ação por skip_frames vezes
        for _ in range(self.skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        return self.process_observation(observation), total_reward, done, info
    
    def process_observation(self, observation):
        """
        Processa uma observação: redimensiona, converte para escala de cinza se necessário
        
        Args:
            observation: Observação original
            
        Returns:
            Observação processada como tensor torch
        """
        import cv2
        
        # Redimensiona para 84x84
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        
        if self.grayscale:
            # Converte para escala de cinza
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            observation = np.expand_dims(observation, axis=0)  # Adiciona dimensão de canal
        else:
            # Reorganiza os canais para (C, H, W)
            observation = np.transpose(observation, (2, 0, 1))
        
        # Normaliza e converte para tensor
        observation = torch.tensor(observation / 255.0, dtype=torch.float32, device=device).unsqueeze(0)
        return observation

# Agente DQN
class DQNAgent:
    def __init__(
        self,
        env,
        buffer_size=100000,
        batch_size=32,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.02,
        eps_decay=100000,
        target_update=10000,
        learning_rate=0.0001
    ):
        """
        Inicializa o agente DQN
        
        Args:
            env: Ambiente pré-processado
            buffer_size: Tamanho do buffer de experiência
            batch_size: Tamanho do batch para treinamento
            gamma: Fator de desconto para recompensas futuras
            eps_start: Epsilon inicial para exploração
            eps_end: Epsilon mínimo
            eps_decay: Taxa de decaimento do epsilon
            target_update: Frequência de atualização da rede target
            learning_rate: Taxa de aprendizado
        """
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.learning_rate = learning_rate
        
        # Obtém o número de ações possíveis
        self.n_actions = self.env.env.action_space.n
        
        # Inicializa as redes neural - principal e target
        self.policy_net = DQN(self.env.observation_shape, self.n_actions).to(device)
        self.target_net = DQN(self.env.observation_shape, self.n_actions).to(device)
        
        # Inicializa os pesos da rede target iguais aos da rede principal
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Coloca a rede target em modo de avaliação
        
        # Inicializa o otimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Inicializa o buffer de experiência
        self.memory = ReplayBuffer(buffer_size)
        
        # Contador de passos para o agente
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """
        Seleciona uma ação baseada no estado atual usando política epsilon-greedy
        
        Args:
            state: Estado atual
            training: Se o agente está em modo de treinamento
            
        Returns:
            Ação selecionada
        """
        # Calcula o epsilon atual
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-self.steps_done / self.eps_decay)
        
        if not training:
            # Durante a avaliação, sempre escolhe a melhor ação
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        
        # Durante o treinamento, usa política epsilon-greedy
        if random.random() > eps_threshold:
            # Escolhe a melhor ação (exploração)
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        else:
            # Escolhe uma ação aleatória (exploração)
            return random.randrange(self.n_actions)
    
    def optimize_model(self):
        """
        Realiza um passo de otimização na rede neural
        """
        # Verifica se o buffer tem amostras suficientes
        if len(self.memory) < self.batch_size:
            return
        
        # Amostra um batch do buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Calcula os valores Q para o estado atual, dado as ações tomadas
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Calcula os valores Q esperados para o próximo estado
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[~dones] = self.target_net(next_states).max(1)[0].detach()[~dones]
        
        # Calcula os valores Q alvo: r + γ * max Q(s', a')
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        
        # Calcula a perda (loss) usando Huber loss (menos sensível a outliers)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Otimiza o modelo
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clipping do gradiente para estabilidade
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes, plot_interval=100):
        """
        Treina o agente por um número específico de episódios
        
        Args:
            num_episodes: Número de episódios para treinar
            plot_interval: Intervalo para plotar o progresso
        """
        # Lista para armazenar as recompensas
        all_rewards = []
        average_rewards = []
        all_losses = []
        
        for episode in range(num_episodes):
            # Reinicia o ambiente
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            done = False
            step_count = 0
            
            while not done:
                # Seleciona uma ação
                action = self.select_action(state)
                
                # Executa a ação
                next_state, reward, done, info = self.env.step(action)
                
                # Armazena a transição no buffer
                self.memory.push(state, action, reward, next_state, done)
                
                # Move para o próximo estado
                state = next_state
                
                # Otimiza o modelo
                loss = self.optimize_model()
                if loss is not None:
                    episode_loss += loss
                
                # Incrementa contadores
                episode_reward += reward
                self.steps_done += 1
                step_count += 1
                
                # Atualiza a rede target periodicamente
                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
                # for debugging
                # print(done, step_count, action, episode_reward, info)

            
            # Registra recompensas e perdas
            all_rewards.append(episode_reward)
            avg_reward = np.mean(all_rewards[-100:])  # Média móvel das últimas 100 recompensas
            average_rewards.append(avg_reward)
            
            if loss is not None:
                all_losses.append(episode_loss / step_count)
            
            # Imprime progresso
            if (episode + 1) % plot_interval == 0:
                clear_output(wait=True)
                print(f"Episódio {episode+1}/{num_episodes}")
                print(f"Recompensa média (últimos 100): {avg_reward:.2f}")
                print(f"Epsilon: {self.eps_end + (self.eps_start - self.eps_end) * np.exp(-self.steps_done / self.eps_decay):.4f}")
                
                # Plota as recompensas
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(all_rewards, label='Recompensa por episódio')
                plt.plot(average_rewards, label='Média móvel (100 ep.)')
                plt.xlabel('Episódio')
                plt.ylabel('Recompensa')
                plt.legend()
                
                if all_losses:
                    plt.subplot(1, 2, 2)
                    plt.plot(all_losses)
                    plt.xlabel('Episódio')
                    plt.ylabel('Perda média')
                
                plt.tight_layout()
                plt.show()
        
        return all_rewards, all_losses
    
    def evaluate(self, num_episodes=10, render=True):
        """
        Avalia o agente treinado
        
        Args:
            num_episodes: Número de episódios para avaliação
            render: Se deve renderizar o ambiente
            
        Returns:
            Recompensa média
        """
        total_rewards = []
        
        a=0 # for debugging
        for episode in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            done = False
            ac = 0
            action = 0
            while not done:
                # Seleciona a melhor ação (sem exploração)
                action_ = self.select_action(state, training=False)
                
                # debugging
                if action_ == action:
                    ac += 1
                if ac > 7:  # se ficar travado dá um tiro!
                    # action = random.randrange(self.n_actions)
                    action = 1 # 1 = FIRE
                    ac = 0
                else:
                    action = action_
                    
                
                # Executa a ação
                next_state, reward, done, info = self.env.step(action)
                
                # if render:
                #     self.env.env.render()
                # Para render_mode='human', o rendering acontece automaticamente
                # Não é necessário chamar env.render() explicitamente
                
                # Atualiza estado e recompensa
                state = next_state
                episode_reward += reward
                
                # for debugging
                print(a, action, episode_reward, info, end='\r')
                a+=1
                # time.sleep(.1)
                            
            total_rewards.append(episode_reward)
            print(f"\nEpisódio {episode+1}: Recompensa = {episode_reward}")
            a=0
        avg_reward = np.mean(total_rewards)
        print(f"Recompensa média: {avg_reward:.2f}")
        return avg_reward
    
    def save_model(self, path):
        """
        Salva o modelo treinado
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
        print(f"Modelo salvo em {path}")
    
    def load_model(self, path):
        """
        Carrega um modelo treinado
        """
        # checkpoint = torch.load(path)
        checkpoint = torch.load(path, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        print(f"Modelo carregado de {path}")


# Função principal para executar o treinamento
# Cria o ambiente Atari Breakout
gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
observation, info = env.reset()

# Aplica o pré-processamento
wrapped_env = AtariPreprocessing(env)

# Cria o agente DQN
agent = DQNAgent(
    wrapped_env,
    buffer_size=100000,
    batch_size=32,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=100000,
    target_update=10000,
    learning_rate=0.0001
)

path = 'C:\\Leo\\python scripts\\Atari Breakout DQN\\'

# Carrega o modelo treinado para continuar o treinamento
# Loading General Checkpoint for Resuming Training
# agent.load_model(path+"breakout_dqn_model.pth")

# Treina o agente
rewards, losses = agent.train(num_episodes=5000, plot_interval=100)

# # Salva o modelo treinado
agent.save_model(path+"breakout_dqn_model.pth")

# Carrega o modelo treinado para avaliação
# Loading Model for inference
agent.load_model(path+"breakout_dqn_model.pth")

# Em seguida, crie um novo ambiente para avaliar com render_mode='human'
env_eval = gym.make('ALE/Breakout-v5', render_mode='human')
observation, info = env_eval.reset()

wrapped_env_eval = AtariPreprocessing(env_eval)

# Crie um novo agente ou atualize o ambiente do agente existente
agent.env = wrapped_env_eval  # Substitua o ambiente no agente
# Ou crie um novo agente
# agent_eval = DQNAgent(wrapped_env_eval, ...)
# agent_eval.load_model("breakout_dqn_model.pth")

# Agora avalie
agent.evaluate(num_episodes=10)

env.close()
pygame.quit()


# Avalia o agente
# env.reset()
# env = gym.make('ALE/Breakout-v5', render_mode='human')
# agent.evaluate(num_episodes=5)


