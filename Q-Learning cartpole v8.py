'''
Cart Pole
=========
This environment is part of the Classic Control environments which contains 
general information about the environment.

Action Space: Discrete(2)
Observation Space: Box([-4.8 -inf -0.41887903 -inf], [4.8 inf 0.41887903 inf], (4,), float32)
import: gymnasium.make("CartPole-v1")

Description
===========
This environment corresponds to the version of the cart-pole problem described 
by Barto, Sutton, and Anderson in “Neuronlike Adaptive Elements That Can Solve 
Difficult Learning Control Problem”. A pole is attached by an un-actuated joint 
to a cart, which moves along a frictionless track. The pendulum is placed upright 
on the cart and the goal is to balance the pole by applying forces in the left and 
right direction on the cart.

Action Space
============
The action is a ndarray with shape (1,) which can take values {0, 1} indicating 
the direction of the fixed force the cart is pushed with.

0: Push cart to the left
1: Push cart to the right

Note: The velocity that is reduced or increased by the applied force is not fixed 
and it depends on the angle the pole is pointing. The center of gravity of the 
pole varies the amount of energy needed to move the cart underneath it

Observation Space
=================
The observation is a ndarray with shape (4,) with the values corresponding to 
the following positions and velocities:

Num    Observation           Min                Max
----------------------------------------------------
0      Cart Position          -4.8               4.8                  
1      Cart Velocity          -Inf               Inf
2      Pole Angle             -0.418 rad (-24°)  0.418 rad (24°)
3      Pole Angular Velocity  -Inf               Inf

Note: While the ranges above denote the possible values for observation space of 
each element, it is not reflective of the allowed values of the state space in an 
unterminated episode. 
Particularly:
The cart x-position (index 0) can be take values between (-4.8, 4.8), but the 
episode terminates if the cart leaves the (-2.4, 2.4) range.
The pole angle can be observed between (-.418, .418) radians (or ±24°), but the 
episode terminates if the pole angle is not in the range (-.2095, .2095) (or ±12°)

Rewards
=======
Since the goal is to keep the pole upright for as long as possible, by default, 
a reward of +1 is given for every step taken, including the termination step. 
The default reward threshold is 500 for v1 and 200 for v0 due to the time limit
 on the environment.

If sutton_barto_reward=True, then a reward of 0 is awarded for every non-terminating
step and -1 for the terminating step. As a result, the reward threshold is 0 for v0 and v1.

Starting State
==============
All observations are assigned a uniformly random value in (-0.05, 0.05)

Episode End
===========
The episode ends if any one of the following occurs:
Termination: Pole Angle is greater than ±12°
Termination: Cart Position is greater than ±2.4 (center of the cart reaches 
the edge of the display)
Truncation: Episode length is greater than 500 (200 for v0)

References
==========
https://gymnasium.farama.org/environments/classic_control/cart_pole/
https://github.com/johnnycode8/gym_solutions/blob/main/cartpole_q.py
https://github.com/LeonimerMelo/Reinforcement-Learning/tree/Q-Learning
'''

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from tqdm import tqdm

# Treinamento --> is_training = True
# Avaliação --> is_training = False
is_training = False
# is_training = True

avaliable_epsodes = 6 # episódios na avaliação do modelo

# if is training no render
render = True
if is_training:
    render = False

# stop episode if rewards reach limit    
# truncate afther this
max_episode_steps_ = 10000
env = gym.make('CartPole-v1', render_mode='human' if render else None,
               max_episode_steps = max_episode_steps_)

# discretização do espaço de estados em [din] amostras para cada estado
# Divide position, velocity, pole angle, and pole angular velocity into segments (bins)
din = 8  # dimensões da Q-table
# episode terminates if the cart leaves the (-2.4, 2.4) range
pos_space = np.linspace(-2.4, 2.4, din)
vel_space = np.linspace(-4, 4, din)
# episode terminates if the pole angle is not in the range (-.2095, .2095) (or ±12°)
ang_space = np.linspace(-.2095, .2095, din)
ang_vel_space = np.linspace(-4, 4, din)

# hiperparâmetros
episodes = 20000
learning_rate_a = 0.2 # alpha or learning rate
discount_factor_g = 0.9 # gamma or discount factor.
epsilon = 1         # 1 = 100% random actions
min_epsilon = 0.01  # Epsilon mínimo
epsilon_decay_rate = 3/episodes # epsilon decay rate

# treinamento
if(is_training):
    # inicializa a Q-table como um tensor de dinxdinxdinxdinx2
    # env.action_space.n --> 2
    #q_table = np.zeros((len(pos_space), len(vel_space), len(ang_space), len(ang_vel_space), env.action_space.n))
    q_table = np.zeros((din, din, din, din, env.action_space.n))
    state_minmax = np.zeros(8)
    steps_per_episode = []
    q_table_history = []   
    rewards_per_episode = []
    for episode in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space) - 1
        state_v = np.digitize(state[1], vel_space) - 1
        state_a = np.digitize(state[2], ang_space) - 1
        state_av = np.digitize(state[3], ang_vel_space) - 1
        rewards = 0
        done = False # True when reached goal
        truncated = False
        time_steps = 0
        while not done and not truncated:
            # escolher a ação (exploration vs. exploitation)
            if np.random.random() < epsilon:
                # Choose random action  (0=go left, 1=go right)
                action = env.action_space.sample() # exploration
            else:
                action = np.argmax(q_table[state_p, state_v, state_a, state_av, :]) # exploitation
    
            # get new state and reward from action
            new_state, reward, done, truncated, _ = env.step(action)
            
            j = 0
            # save global states MIN and MAX
            for s in range(4):
                state_minmax[j] = min(state_minmax[j], new_state[s])
                j += 1
                state_minmax[j] = max(state_minmax[j], new_state[s])
                j += 1
 
            new_state_p = np.digitize(new_state[0], pos_space) - 1
            new_state_v = np.digitize(new_state[1], vel_space) - 1
            new_state_a = np.digitize(new_state[2], ang_space) - 1
            new_state_av= np.digitize(new_state[3], ang_vel_space) - 1
    
            if is_training:
                q_table[state_p, state_v, state_a, state_av, action] = q_table[state_p, state_v, state_a, state_av, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q_table[new_state_p, new_state_v, new_state_a, new_state_av,:]) - q_table[state_p, state_v, state_a, state_av, action]
                )
    
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av
    
            rewards += reward
            time_steps += 1
    
        q_table_history.append(np.mean(q_table))  # Armazenar média geral da Q-Table
        steps_per_episode.append(time_steps)
        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
        t = 100 * (episode+1) / episodes
        print('episode: %d \ttotal: %.1f%% \trewards(time steps): %d     \tepsilon: %.2f \tmean rewards: %0.1f    ' % 
              (episode+1, t, time_steps, epsilon, mean_rewards), end='\r')
        
        # Decaindo o epsilon
        a = epsilon * epsilon_decay_rate
        b = epsilon - a
        epsilon = max(b, min_epsilon)  
        # epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        # if episode > (episodes - 10):
        #     env = gym.make('CartPole-v1', render_mode='human')
    
    # Save Q table to file
    f = open('cartpole_'+str(episodes)+'_'+str(learning_rate_a)+'_'+str(din)+'.pkl','wb')
    pickle.dump(q_table, f)
    f.close()
    
    print('\n')
    print('state:      [MIN,   MAX ]')
    print('=========================')
    print('posição:    [%.2f, %.2f]' % (state_minmax[0], state_minmax[1]))
    print('velocidade: [%.2f, %.2f]' % (state_minmax[2], state_minmax[3]))
    print('pos. ang.:  [%.2f, %.2f]' % (state_minmax[4], state_minmax[5]))
    print('vel. ang.:  [%.2f, %.2f]' % (state_minmax[6], state_minmax[7]))
    print('\n')
    
    # Metrics after training
    mean_rewards = np.zeros(episodes)
    e = episodes//100
    for t in tqdm(range(episodes)):
        # calculo a média móvel dos rewards de 100 episódios
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-e):(t+1)])
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
    
# Avaliação
if not is_training:
    tst = True
    while tst:
        try:
            # Read trained Q-table from file
            f = open('cartpole_'+str(episodes)+'_'+str(learning_rate_a)+'_'+str(din)+'.pkl', 'rb')
            q_table = pickle.load(f)
            f.close()
        except:
            print('arquivo não encontrado! Você deve treinar o modelo primeiro!')
            break
            
        env.reset()
        env.render()
        max_reward = 0
        for eps in range(avaliable_epsodes):
            state = env.reset()[0]      # Starting position, starting velocity always 0
            state_p = np.digitize(state[0], pos_space) - 1
            state_v = np.digitize(state[1], vel_space) - 1
            state_a = np.digitize(state[2], ang_space) - 1
            state_av = np.digitize(state[3], ang_vel_space) - 1
            
            done = False
            time_steps_ = 0
            rw = 0
            truncated = False
            while not done and not truncated:
                action = np.argmax(q_table[state_p, state_v, state_a, state_av, :])
                new_state, reward, done, truncated, _ = env.step(action)
                new_state_p = np.digitize(new_state[0], pos_space) - 1
                new_state_v = np.digitize(new_state[1], vel_space) - 1
                new_state_a = np.digitize(new_state[2], ang_space) - 1
                new_state_av= np.digitize(new_state[3], ang_vel_space) - 1
                
                state = new_state
                state_p = new_state_p
                state_v = new_state_v
                state_a = new_state_a
                state_av = new_state_av
        
                time_steps_ += 1
                rw += reward
            
            max_reward = max(max_reward, rw)
            t = 100 * eps / avaliable_epsodes
            print('episode: %d \ttotal: %.1f%% \trewards(time steps): %d   \tmax rewards: %d    ' % 
                  (eps+1, t, rw, max_reward), end='\r')
            
        tst = False

env.close()




