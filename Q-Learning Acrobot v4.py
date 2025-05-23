"""
    ### Description

    The Acrobot environment is based on Sutton's work in
    ["Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)
    and [Sutton and Barto's book](http://www.incompleteideas.net/book/the-book-2nd.html).
    The system consists of two links connected linearly to form a chain, with one end of
    the chain fixed. The joint between the two links is actuated. The goal is to apply
    torques on the actuated joint to swing the free end of the linear chain above a
    given height while starting from the initial state of hanging downwards.

    As seen in the **Gif**: two blue links connected by two green joints. The joint in
    between the two links is actuated. The goal is to swing the free end of the outer-link
    to reach the target height (black horizontal line above system) by applying torque on
    the actuator.

    ### Action Space

    The action is discrete, deterministic, and represents the torque applied on the actuated
    joint between the two links.

    | Num | Action                                | Unit         |
    |-----|---------------------------------------|--------------|
    | 0   | apply -1 torque to the actuated joint | torque (N m) |
    | 1   | apply 0 torque to the actuated joint  | torque (N m) |
    | 2   | apply 1 torque to the actuated joint  | torque (N m) |

    ### Observation Space

    The observation is a `ndarray` with shape `(6,)` that provides information about the
    two rotational joint angles as well as their angular velocities:

    | Num | Observation                  | Min                 | Max               |
    |-----|------------------------------|---------------------|-------------------|
    | 0   | Cosine of `theta1`           | -1                  | 1                 |
    | 1   | Sine of `theta1`             | -1                  | 1                 |
    | 2   | Cosine of `theta2`           | -1                  | 1                 |
    | 3   | Sine of `theta2`             | -1                  | 1                 |
    | 4   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
    | 5   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |

    where
    - `theta1` is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
    downwards.
    - `theta2` is ***relative to the angle of the first link.***
        An angle of 0 corresponds to having the same angle between the two links.

    The angular velocities of `theta1` and `theta2` are bounded at ±4π, and ±9π rad/s respectively.
    A state of `[1, 0, 1, 0, ..., ...]` indicates that both links are pointing downwards.

    ### Rewards

    The goal is to have the free end reach a designated target height in as few steps as possible,
    and as such all steps that do not reach the goal incur a reward of -1.
    Achieving the target height results in termination with a reward of 0. The reward threshold is -100.

    ### Starting State

    Each parameter in the underlying state (`theta1`, `theta2`, and the two angular velocities) is initialized
    uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.

    ### Episode End

    The episode ends if one of the following occurs:
    1. Termination: The free end reaches the target height, which is constructed as:
    `-cos(theta1) - cos(theta2 + theta1) > 1.0`
    2. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments

    No additional arguments are currently supported.

    ```
    env = gym.make('Acrobot-v1')
    ```

    By default, the dynamics of the acrobot follow those described in Sutton and Barto's book
    [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/11/node4.html).
    However, a `book_or_nips` parameter can be modified to change the pendulum dynamics to those described
    in the original [NeurIPS paper](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html).

    ```
    # To change the dynamics as described above
    env.env.book_or_nips = 'nips'
    ```

    See the following note and
    the [implementation](https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py) for details:

    > The dynamics equations were missing some terms in the NIPS paper which
            are present in the book. R. Sutton confirmed in personal correspondence
            that the experimental results shown in the paper and the book were
            generated with the equations shown in the book.
            However, there is the option to run the domain with the paper equations
            by setting `book_or_nips = 'nips'`


    ### Version History

    - v1: Maximum number of steps increased from 200 to 500. The observation space for v0 provided direct readings of
    `theta1` and `theta2` in radians, having a range of `[-pi, pi]`. The v1 observation space as described here provides the
    sine and cosine of each angle instead.
    - v0: Initial versions release (1.0.0) (removed from gym for v1)

    ### References
    - Sutton, R. S. (1996). Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding.
        In D. Touretzky, M. C. Mozer, & M. Hasselmo (Eds.), Advances in Neural Information Processing Systems (Vol. 8).
        MIT Press. https://proceedings.neurips.cc/paper/1995/file/8f1d43620bc6bb580df6e80b0dc05c48-Paper.pdf
    - Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press.
    """

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Treinamento --> is_training = True
# Avaliação --> is_training = False
is_training = False
# is_training = True

avaliable_epsodes = 6 # episódios na avaliação do modelo
max_episode_steps_ = 1000 # truncate afther this
env = gym.make('Acrobot-v1', render_mode='human' if not is_training else None, # if is training no render
               max_episode_steps = max_episode_steps_)

# hyperparameters
episodes = 70000
learning_rate_a = 0.15        # alpha aka learning rate
discount_factor_g = 0.99      # gamma aka discount factor.
epsilon = 1                  # start episilon at 1 (100% random actions)
epsilon_min = 0.01           # minimum epsilon
epsilon_decay_rate = 5/episodes # epsilon decay rate
divisions = 15               # used to convert continuous state space to discrete space

# Divide continuous observation space into discrete segments
th1_cos  = np.linspace(env.observation_space.low[0], env.observation_space.high[0], divisions)
th1_sin  = np.linspace(env.observation_space.low[1], env.observation_space.high[1], divisions)
th2_cos  = np.linspace(env.observation_space.low[2], env.observation_space.high[2], divisions)
th2_sin  = np.linspace(env.observation_space.low[3], env.observation_space.high[3], divisions)
th1_w    = np.linspace(env.observation_space.low[4], env.observation_space.high[4], divisions)
th2_w    = np.linspace(env.observation_space.low[5], env.observation_space.high[5], divisions)
divisions_plus = divisions + 1

if(is_training):
    # initialize q table to 16x16x16x16x16x16x3 array. 
    # 16 divisions of 15 plus 1 extra slice becouse digitize criates divisions + 1 dimensions
    q = np.zeros((divisions_plus, divisions_plus,
                  divisions_plus, divisions_plus,
                  divisions_plus, divisions_plus,
                  env.action_space.n))

    rewards_per_episode = []     # list to store rewards for each episode  
    q_table_history = []  
    epsilon_hist = []
    max_reward = -100000
    for episode in range(episodes):
        state, info = env.reset()      # Starting position
    
        # Convert continuous state to discrete state
        s_i0  = np.digitize(state[0], th1_cos)
        s_i1  = np.digitize(state[1], th1_sin)
        s_i2  = np.digitize(state[2], th2_cos)
        s_i3  = np.digitize(state[3], th2_sin)
        s_i4  = np.digitize(state[4], th1_w)
        s_i5  = np.digitize(state[5], th2_w)
    
        terminated = False          # True when reached goal
        truncated = False
        done = False
        rewards = 0                   # rewards collected per episode
        while not done:
            if np.random.rand() < epsilon:
                # Choose random action
                action = env.action_space.sample()
            else:
                # Choose best action
                action = np.argmax(q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, :])
    
            # Take action
            new_state, reward, terminated, truncated, info = env.step(action)
    
            # Convert continuous state to discrete space
            ns_i0  = np.digitize(new_state[0], th1_cos)
            ns_i1  = np.digitize(new_state[1], th1_sin)
            ns_i2  = np.digitize(new_state[2], th2_cos)
            ns_i3  = np.digitize(new_state[3], th2_sin)
            ns_i4  = np.digitize(new_state[4], th1_w)
            ns_i5  = np.digitize(new_state[5], th2_w)
    
            # Update Q table
            q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, action] += \
                learning_rate_a * (
                    reward +
                    discount_factor_g * np.max(q[ns_i0, ns_i1, ns_i2, ns_i3, ns_i4, ns_i5,:]) -
                    q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, action]
                )

            # Set state to new state
            state = new_state
            s_i0 = ns_i0
            s_i1 = ns_i1
            s_i2 = ns_i2
            s_i3 = ns_i3
            s_i4 = ns_i4
            s_i5 = ns_i5
    
            # Collect rewards
            rewards += reward
            done = terminated or truncated
      
        rewards_per_episode.append(rewards) # Store rewards per episode
        q_table_history.append(np.mean(q))  # Armazenar média geral da Q-Table
        max_reward = max(max_reward, rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:])
        t = 100 * (episode+1) / episodes
        print("\033[K", end="\r") # clear current print line
        print('episode:%d total:%.1f%% epsilon:%.2f rewards:%d mean rewards:%0.1f max reward:%d' % 
              (episode+1, t, epsilon, rewards, mean_rewards, max_reward), end='\r')    
    
        # Decay epsilon
        epsilon = max(epsilon * (1 - epsilon_decay_rate), epsilon_min)  
        epsilon_hist.append(epsilon)
 
    # save trained Q-Table
    f = open('acrobot_'+str(episodes)+'.pkl','wb')
    pickle.dump(q, f)
    f.close()
    
    # Graph mean rewards
    mean_rewards_ = []
    for t in range(episodes):
        # calculo a média móvel dos rewards de 100 episódios
        mean_rewards_.append(np.mean(rewards_per_episode[max(0, t-150):(t+1)]))
    plt.title('Mean rewards per episode')
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.plot(mean_rewards_)
    plt.show()
    
    # Plotar o decaimento do epsilon ao longo dos epsódios
    plt.plot(epsilon_hist, color="red")
    plt.title("Decaimento do epsilon")
    plt.xlabel("Episódios")
    plt.ylabel("epsilon")
    plt.grid(True)
    plt.show()
              
    # Plotar a evolução da média geral da Q-Table
    plt.figure(figsize=(8, 6))
    plt.plot(q_table_history, label="Média Geral da Q-Table", color="blue")
    plt.title("Evolução da Média Geral da Q-Table")
    plt.xlabel("Episódios")
    plt.ylabel("Média dos Valores Q")
    plt.grid(True)
    plt.show()
    
# Avaliação
else:
    tst = True
    while tst:
        try:
            # Read trained Q-table from file
            f = open('acrobot_'+str(episodes)+'.pkl', 'rb')
            q = pickle.load(f)
            f.close()
        except:
            print('arquivo não encontrado! Você deve treinar o modelo primeiro!')
            break
        
        max_reward = -50000
        for eps in range(avaliable_epsodes):
            state, info = env.reset()      # Starting position  
            # Convert continuous state to discrete state
            s_i0  = np.digitize(state[0], th1_cos)
            s_i1  = np.digitize(state[1], th1_sin)
            s_i2  = np.digitize(state[2], th2_cos)
            s_i3  = np.digitize(state[3], th2_sin)
            s_i4  = np.digitize(state[4], th1_w)
            s_i5  = np.digitize(state[5], th2_w)
            rewards = 0
            steps = 0
            terminated = False          # True when reached goal
            truncated = False
            done = False
            while not done:
                action = np.argmax(q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, :]) # exploitation
                
                new_state, reward, terminated, truncated, info = env.step(action)
                
                # Convert continuous state to discrete space
                ns_i0  = np.digitize(new_state[0], th1_cos)
                ns_i1  = np.digitize(new_state[1], th1_sin)
                ns_i2  = np.digitize(new_state[2], th2_cos)
                ns_i3  = np.digitize(new_state[3], th2_sin)
                ns_i4  = np.digitize(new_state[4], th1_w)
                ns_i5  = np.digitize(new_state[5], th2_w)
                
                # Set state to new state
                state = new_state
                s_i0 = ns_i0
                s_i1 = ns_i1
                s_i2 = ns_i2
                s_i3 = ns_i3
                s_i4 = ns_i4
                s_i5 = ns_i5
        
                rewards += reward
                steps += 1
                done = terminated or truncated
                
                # max_reward = max(max_reward, rewards)
                t = 100 * (eps+1) / avaliable_epsodes
                print("\033[K", end="\r") # clear current print line
                # print('                                                                                    ', end='\r')
                print('episode: %d total: %.1f%% time steps: %d rewards: %d max reward: %d' % 
                      (eps+1, t, steps, rewards, max_reward), end='\r') 
            max_reward = max(max_reward, rewards)
        tst = False 
    
env.close()



