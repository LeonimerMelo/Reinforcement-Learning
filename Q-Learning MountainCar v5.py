import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Treinamento --> is_training = True
# Avaliação --> is_training = False
is_training = False
# is_training = True

# if is training no render
render = True
if is_training:
    render = False

max_episode_steps_ = 300
env = gym.make('MountainCar-v0', render_mode='human' if render else None, 
               max_episode_steps = max_episode_steps_,)

# Divide position and velocity into segments
pos_space = np.linspace(env.observation_space.low[0], 
                        env.observation_space.high[0], 20) # Between -1.2 and 0.6
vel_space = np.linspace(env.observation_space.low[1], 
                        env.observation_space.high[1], 20) # Between -0.07 and 0.07

if(is_training):
    q_table = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
else:
    f = open('mountain_car.pkl', 'rb')
    q_table = pickle.load(f)
    f.close()
    
# Hiperparâmetros
episodes = 100  # Número de episódios
alpha = 0.9  # Taxa de aprendizado (alpha or learning rate)
# gamma or discount rate. Near 0: more weight/reward placed on immediate state. 
# Near 1: more on future state
gamma = 0.9  # Fator de desconto (discount factor)
epsilon = 1  # Taxa de exploração inicial (1 = 100% random actions)
#epsilon_decay_rate = 0.001  # Fator de decaimento do epsilon
min_epsilon = 0.01  # Epsilon mínimo
epsilon_decay_rate = 2/episodes # Fator de decaimento do epsilon (epsilon decay rate)

if(is_training):
    rewards_per_episode = np.zeros(episodes)
    for episode in range(episodes):
        state = env.reset()[0]      # Starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        terminated = False          # True when reached goal
        rewards = 0
        while(not terminated and rewards > -1000):
            # random number generator (0.0,1.0)
            if np.random.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_p, state_v, :])
    
            new_state,reward,terminated,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
    
            q_table[state_p, state_v, action] = q_table[state_p, state_v, action] + alpha * (
                reward + gamma * np.max(
                    q_table[new_state_p, new_state_v,:]
                    ) - q_table[state_p, state_v, action]
                )
    
            # state = new_state
            state_p = new_state_p
            state_v = new_state_v
    
            rewards += reward
    
        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)
        rewards_per_episode[episode] = rewards
        t = 100 * episode / episodes
        print('episode: %d \ttotal: %.1f%% \tepsilon: %.2f \trewards: %d' % 
              (episode, t, epsilon, rewards), end='\r')
        
    f = open('mountain_car.pkl','wb') # Save Q-table to file
    pickle.dump(q_table, f)
    f.close()
    print("\nTreinamento concluído!")
    
print("Tabela Q:")
print(q_table[0])

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
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(mean_rewards)
    plt.show()
    #plt.savefig(f"mountain_car.png")

