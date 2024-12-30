'''
Mountain Car
============
This environment is part of the Classic Control environments which contains 
general information about the environment.

Action Space: Discrete(3)
Observation Space: Box([-1.2 -0.07], [0.6 0.07], (2,), float32)
import: gymnasium.make("MountainCar-v0")

Description
===========
The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically 
at the bottom of a sinusoidal valley, with the only possible actions being the 
accelerations that can be applied to the car in either direction. The goal of 
the MDP is to strategically accelerate the car to reach the goal state on top 
of the right hill. There are two versions of the mountain car domain in gymnasium: 
one with discrete actions and one with continuous. This version is the one with 
discrete actions.

Observation Space
=================
The observation is a ndarray with shape (2,) where the elements correspond to 
the following:
Num  Observation                            Min    Max   Unit
-------------------------------------------------------------
0    position of the car along the x-axis   -1.2   0.6   position (m)
1    velocity of the car                    -0.07  0.07  velocity (v)

Action Space
============
There are 3 discrete deterministic actions:
0: Accelerate to the left
1: Don’t accelerate
2: Accelerate to the right

Reward
======
The goal is to reach the flag placed on top of the right hill as quickly as 
possible, as such the agent is penalised with a reward of -1 for each timestep.

Starting State
==============
The position of the car is assigned a uniform random value in [-0.6 , -0.4]. 
The starting velocity of the car is always assigned to 0.

Episode End
===========
The episode ends if either of the following happens:
Termination: The position of the car is greater than or equal to 0.5 
(the goal position on top of the right hill)
Truncation: The length of the episode is 200 (original max episode steps).

https://gymnasium.farama.org/environments/classic_control/mountain_car/
'''

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

# Treinamento --> is_training = True
# Avaliação --> is_training = False
is_training = False
is_training = True

# if is training no render
render = True
if is_training:
    render = False

max_episode_steps_ = 1000 # truncate afther this
env = gym.make('MountainCar-v0', render_mode='human' if render else None, 
               max_episode_steps = max_episode_steps_,)

# Divide position and velocity into 20 discretized segments
pos_space = np.linspace(env.observation_space.low[0], 
                        env.observation_space.high[0], 20) # Between -1.2 and 0.6
vel_space = np.linspace(env.observation_space.low[1], 
                        env.observation_space.high[1], 20) # Between -0.07 and 0.07

if(is_training):
    # a tabela Q-table é um tensor 20x20x3
    # matriz tridimensional, onde o eixo x = posição (pos_space)
    # o eixo y = velocidade (vel_space) e o eixo z = action (0, 1 ou 2)
    q_table = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
else:
    f = open('mountain_car.pkl', 'rb')
    q_table = pickle.load(f)
    f.close()
    
# Hiperparâmetros
episodes = 3000  # Número de episódios
alpha = 0.1  # Taxa de aprendizado (alpha or learning rate)
# gamma or discount rate. Near 0: more weight/reward placed on immediate state. 
# Near 1: more on future state
gamma = 0.9  # Fator de desconto (discount factor)
epsilon = 1  # Taxa de exploração inicial (1 = 100% random actions)
#epsilon_decay_rate = 0.001  # Fator de decaimento do epsilon
min_epsilon = 0.01  # Epsilon mínimo
epsilon_decay_rate = 3/episodes # Fator de decaimento do epsilon (epsilon decay rate)

if(is_training):
    rewards_per_episode = np.zeros(episodes)
    steps_per_episode = []
    q_table_history = []
    for episode in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        # retorna o índice da posição inicial [-0.6 , -0.4] dentro da tabela pos_space (de 0 a 19)
        i_state_p = np.digitize(state[0], pos_space)
        # retorna o índice da velocidade inicial (= 0) dentro da tabela vel_space (de 1 a 19)
        i_state_v = np.digitize(state[1], vel_space)
        done = False          # True when reached goal
        truncated = False
        rewards = 0
        time_steps = 0
        while not done and not truncated:
            # random number generator (0.0,1.0)
            if np.random.random() < epsilon:
                # Choose random action (0: Accelerate to the left, 
                # 1: Don’t accelerate, 2: Accelerate to the right)
                action = env.action_space.sample()
            else:
                # action = índice com o valor máximo da tabela Q-table
                # entre as 3 ações possíveis (eixo z) estando o agente no estado:
                # x = i_state_p (posição) e y = i_state_v (velocidade)
                action = np.argmax(q_table[i_state_p, i_state_v, :])
    
            new_state, reward, done, truncated ,_ = env.step(action)
            # retorna o índice da nova posição dentro da tabela pos_space (de 0 a 19)
            i_new_state_p = np.digitize(new_state[0], pos_space)
            # retorna o índice da nova velocidade dentro da tabela vel_space (de 1 a 19)
            i_new_state_v = np.digitize(new_state[1], vel_space)
    
            # atualizo a Q-table
            q_table[i_state_p, i_state_v, action] = (
                q_table[i_state_p, i_state_v, action] + alpha  * (
                    reward + gamma * np.max(q_table[i_new_state_p, i_new_state_v,:]) 
                    - q_table[i_state_p, i_state_v, action]))
    
            # state = new_state
            i_state_p = i_new_state_p
            i_state_v = i_new_state_v
    
            rewards += reward
            time_steps += 1
    
        # Decaindo o epsilon
        a = epsilon * epsilon_decay_rate
        b = epsilon - a
        epsilon = max(b, min_epsilon)     
        #epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)
        
        q_table_history.append(np.mean(q_table))  # Armazenar média geral da Q-Table
        steps_per_episode.append(time_steps)
        rewards_per_episode[episode] = rewards
        t = 100 * episode / episodes
        print('episode: %d \ttotal: %.1f%% \tepsilon: %.2f \trewards: %d  \ttime steps: %d      '% 
              (episode, t, epsilon, rewards, time_steps), end='\r')
        
    f = open('mountain_car.pkl','wb') # Save Q-table to file
    pickle.dump(q_table, f)
    f.close()
    print("\nTreinamento concluído!")
    
# print("Tabela Q:")
# print(q_table[2])

# Avaliação
if not is_training:
    state = env.reset()[0] # Starting position, starting velocity always 0
    new_state_p = np.digitize(state[0], pos_space)
    new_state_v = np.digitize(state[1], vel_space)
    done = False
    env.render()
    i=0
    rw=0
    truncated = False
    while not done and not truncated:
        action = np.argmax(q_table[new_state_p, new_state_v, :])
        next_state, reward, done, truncated, _ = env.step(action)
        new_state_p = np.digitize(next_state[0], pos_space)
        new_state_v = np.digitize(next_state[1], vel_space)

        i+=1
        print('step time:', i, end='\r')
        rw+=reward
    
    print('\nreward:', rw)

env.close()

# Metrics after training
if is_training:
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        # calculo a média móvel dos rewards de 100 episódios
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.title('Mean rewards per episode')
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.plot(mean_rewards)
    plt.show()

    # Gráfico Total steps per episode
    plt.title('Total time steps per episode')
    plt.xlabel('time steps')
    plt.ylabel('total')
    sns.histplot(steps_per_episode, bins=40, kde=True)
    plt.show()
    
    # Plotar a evolução da média geral da Q-Table
    plt.figure(figsize=(8, 6))
    plt.plot(q_table_history, label="Média Geral da Q-Table", color="blue")
    plt.title("Evolução da Média Geral da Q-Table")
    plt.xlabel("Episódios")
    plt.ylabel("Média dos Valores Q")
    #plt.legend()
    plt.grid(True)
    plt.show()