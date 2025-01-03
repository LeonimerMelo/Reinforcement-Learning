'''
Pendulum
========
O ambiente Pendulum-v1 tem um espaço de ações contínuo, mas o Q-learning tradicional 
é projetado para ambientes com espaços de ações discretos. Portanto, para aplicar Q-learning 
no Pendulum-v1, é necessário fazer uma discretização dos espaços de estados e ações.

Description
===========
The inverted pendulum swingup problem is based on the classic problem in control theory.
The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
into an upright position, with its center of gravity right above the fixed point.

The diagram below specifies the coordinate system used for the implementation of the pendulum's
dynamic equations.

-  `x-y`: cartesian coordinates of the pendulum's end in meters.
- `theta` : angle in radians.
- `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

Action Space
============
The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

| Num | Action | Min  | Max |
|-----|--------|------|-----|
| 0   | Torque | -2.0 | 2.0 |

Observation Space
=================
The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
end and its angular velocity.

| Num | Observation      | Min  | Max |
|-----|------------------|------|-----|
| 0   | x = cos(theta)   | -1.0 | 1.0 |
| 1   | y = sin(theta)   | -1.0 | 1.0 |
| 2   | Angular Velocity | -8.0 | 8.0 |

Rewards
=======
The reward function is defined as:
r = -(theta2 + 0.1 * theta_dt2 + 0.001 * torque2)
where theta is the pendulum’s angle normalized between [-pi, pi] (with 0 being 
in the upright position). Based on the above equation, the minimum reward that
can be obtained is -(pi2 + 0.1 * 82 + 0.001 * 22) = -16.2736044, while the maximum 
reward is zero (pendulum is upright with zero velocity and no torque applied).

Starting State
==============
The starting state is a random angle in [-pi, pi] and a random angular velocity in [-1,1].

Episode Truncation
==================
The episode truncates at 200 time steps (by default).

References
==========
https://gymnasium.farama.org/environments/classic_control/pendulum/
https://github.com/johnnycode8/gym_solutions/blob/main/pendulum_q.py
https://github.com/LeonimerMelo/Reinforcement-Learning/tree/Q-Learning
'''

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Treinamento --> is_training = True
# Avaliação --> is_training = False
is_training = False
is_training = True

avaliable_epsodes = 6 # episódios na avaliação do modelo

# if is training no render
render = True
if is_training:
    render = False
    
max_episode_steps_ = 500 # truncate afther this
env = gym.make('Pendulum-v1', render_mode='human' if render else None,
               max_episode_steps = max_episode_steps_)

# hyperparameters
episodes = 150
learning_rate_a = 0.1        # alpha aka learning rate
discount_factor_g = 0.9      # gamma aka discount factor.
epsilon = 1                  # start episilon at 1 (100% random actions)
# epsilon_decay_rate = 0.0005  # epsilon decay rate
epsilon_decay_rate = 3/episodes # epsilon decay rate
epsilon_min = 0.05           # minimum epsilon

# discretização do espaço de estados em [din] amostras para cada estado
din = 15  # used to convert continuous state space to discrete space
# Divide observation space into discrete segments
x  = np.linspace(env.observation_space.low[0], env.observation_space.high[0], din)
y  = np.linspace(env.observation_space.low[1], env.observation_space.high[1], din)
w  = np.linspace(env.observation_space.low[2], env.observation_space.high[2], din)

# Divide action space into discrete segments
# discretização das ações em [din] amostras
a = np.linspace(env.action_space.low[0], env.action_space.high[0], din)

if(is_training):
    # initialize q table to 16x16x16x16 array if din = 15  
    q = np.zeros((len(x)+1, len(y)+1, len(w)+1, len(a)+1))
    best_reward = -99999
    rewards_per_episode = []     # list to store rewards for each episode
    i = 0
    for episode in range(episodes):
        # The starting state is a random angle in [-pi, pi] and a random angular velocity in [-1,1].
        state = env.reset()[0]      
        s_i0  = np.digitize(state[0], x)
        s_i1  = np.digitize(state[1], y)
        s_i2  = np.digitize(state[2], w)
    
        rewards = 0
        steps = 0
        done = False 
        truncated = False
        while not done and not truncated:
            if np.random.rand() < epsilon:
                # Choose random action
                action = env.action_space.sample()
                action_idx = np.digitize(action, a)
            else:
                action_idx = np.argmax(q[s_i0, s_i1, s_i2, :])
                action = a[action_idx-1]
                action = np.array([action])
    
            # Take action
            new_state, reward, done, truncated ,_ = env.step(action)
    
            # Discretize new state
            ns_i0  = np.digitize(new_state[0], x)
            ns_i1  = np.digitize(new_state[1], y)
            ns_i2  = np.digitize(new_state[2], w)
    
            # Update Q table
            if is_training:
                q[s_i0, s_i1, s_i2, action_idx] = \
                    q[s_i0, s_i1, s_i2, action_idx] + \
                    learning_rate_a * (
                        reward + discount_factor_g*np.max(q[ns_i0, ns_i1, ns_i2,:])
                            - q[s_i0, s_i1, s_i2, action_idx]
                    )
    
            state = new_state
            s_i0 = ns_i0
            s_i1 = ns_i1
            s_i2 = ns_i2
    
            rewards += reward
            steps += 1
    
            if rewards>best_reward:
                best_reward = rewards
                # Save Q table to file on new best reward
                f = open('pendulum_'+str(episodes)+'.pkl','wb')
                pickle.dump(q, f)
                f.close()
    
        # Store rewards per episode
        rewards_per_episode.append(rewards)
    
        # Print stats
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
        t = 100 * (episode+1) / episodes
        print('episode: %d \ttotal: %.1f%% \ttime steps: %d \tepsilon: %.2f \trewards: %d \tmean rewards: %0.1f \tbest reward: %d' % 
              (episode+1, t, steps, epsilon, rewards, mean_rewards, best_reward), end='\r') 
    
        # Decaindo o epsilon
        k = epsilon - epsilon * epsilon_decay_rate
        epsilon = max(k, epsilon_min) 

    

if is_training:
    # Graph mean rewards
    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
    plt.plot(mean_rewards)
    plt.show()
    
if not is_training:
    f = open('pendulum_'+str(episodes)+'.pkl', 'rb')
    q = pickle.load(f)
    f.close()


env.close()



