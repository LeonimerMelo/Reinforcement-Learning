# https://github.com/jiseongHAN/Super-Mario-RL/blob/master/README.md

# pip install numpy==1.26.4
# pip install torch>=1.6.0
# pip install torchvision
# pip install gym==0.23
# pip install nes-py
# pip install gym-super-mario-bros==7.2.3
# pip install opencv-python
# pip install matplotlib
# pip install -r requirements.txt

# import sys
import matplotlib.pyplot as plt
# import pickle
import random
from collections import deque # https://www.geeksforgeeks.org/deque-in-python/
import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import time
# import os
import cv2
import gym
from gym import spaces
from gym.wrappers import TimeLimit
import pygame
import ale_py
import shimmy

# C:\Users\TechnoLEDs
# os.environ.setdefault("PATH", "")
cv2.ocl.setUseOpenCL(False)
path='C:\\Leo\\python scripts\\Super Mario DQN\\'

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    # Putting *args and/or **kwargs as the last items in your function definition’s 
    # argument list allows that function to accept an arbitrary number of arguments 
    # and/or keyword arguments.
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind(
    env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True
):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class EpisodicLifeMario(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._life
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped._life
        return obs


def wrap_mario(env):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeMario(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    # env = custom_reward(env)
    env = FrameStack(env, 4)
    return env

def arange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)


class replay_memory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)


class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)

        self.seq.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.max(-1, True)[0])

        return q


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train(q, q_target, memory, batch_size, gamma, optimizer, device):
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    s = np.array(s).squeeze()
    s_prime = np.array(s_prime).squeeze()
    a_max = q(s_prime).max(1)[1].unsqueeze(-1)
    r = torch.FloatTensor(r).unsqueeze(-1).to(device)
    done = torch.FloatTensor(done).unsqueeze(-1).to(device)
    with torch.no_grad():
        y = r + gamma * q_target(s_prime).gather(1, a_max) * done
    a = torch.tensor(a).unsqueeze(-1).to(device)
    q_value = torch.gather(q(s), dim=1, index=a.view(-1, 1).long())

    # Calcular a perda e atualizar a rede
    loss = F.smooth_l1_loss(q_value, y).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, optimizer

def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)

score_lst = []
loss_lst = []   
def main(epochs, env, q, q_target, optimizer, device):
    t = 0
    gamma = 0.99
    batch_size = 256
    optimizer_ = optimizer
    
    N = 50000
    # eps = 0.001
    
    EPSILON_START = 1.0
    EPSILON_END = 0.001
    EPSILON_DECAY = 0.95
    epsilon = EPSILON_START
    
    memory = replay_memory(N)
    update_interval = 50
    print_interval = 1

    # score_lst = []
    total_score = 0.0
    loss = 0.0
    start_time = time.perf_counter()

    for k in range(epochs):
        s = arange(env.reset())
        done = False

        while not done:
            
            if epsilon > np.random.rand():
                a = env.action_space.sample()
            else:
                if device == "cpu":
                    a = np.argmax(q(s).detach().numpy())
                else:
                    a = np.argmax(q(s).cpu().detach().numpy())
            s_prime, r, done, _ = env.step(a)
            s_prime = arange(s_prime)
            total_score += r
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
            memory.push((s, float(r), int(a), s_prime, int(1 - done)))
            s = s_prime
            stage = env.unwrapped._stage
            
            
            if len(memory) > 2000:
                loss_, optimizer_ = train(q, q_target, memory, batch_size, gamma, optimizer, device)
                # loss += train(q, q_target, memory, batch_size, gamma, optimizer, device)
                loss += loss_
                t += 1
                
            if t % update_interval == 0:
                copy_weights(q, q_target)
                # torch.save(q.state_dict(), path + "mario_q.pth")
                # torch.save(q_target.state_dict(), path + 'mario_q_target.pth')
                  
        if k % print_interval == 0:
            time_spent, start_time = time.perf_counter() - start_time, time.perf_counter()
            print(
                "%s | Epoch: %d | Epsilon: %.3f | score: %d | loss: %.1f | stage: %d | time spent: %.1f"
                % (
                    device,
                    k+1,
                    epsilon,
                    total_score / print_interval,
                    loss / print_interval,
                    stage,
                    time_spent,
                )
            )
            loss_lst.append(loss / print_interval)
            score_lst.append(total_score / print_interval)
            total_score = 0
            loss = 0.0
            # pickle.dump(score_lst, open("score.p", "wb"))
         
        # Decaimento do epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            
        # save models
        torch.save(q.state_dict(), path + "mario_q.pth")
        torch.save(q_target.state_dict(), path + 'mario_q_target.pth')
        
        # Saving General Checkpoint for Resuming Training
        torch.save({
                    'epoch': k,
                    'q_state_dict': q.state_dict(),
                    'q_target_state_dict': q_target.state_dict(),
                    'optimizer_state_dict': optimizer_.state_dict(),
                    'loss': loss,
                    #'acc': n_acc,
                    }, path + "model_mario_train.pth")


def train_agent(epochs=10000, resuming_training=False):
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(device)
    
    # q models
    q = model(n_frame, env.action_space.n, device).to(device)
    q_target = model(n_frame, env.action_space.n, device).to(device)
    optimizer = optim.Adam(q.parameters(), lr=0.0001)
    
    if resuming_training:
        '''
        Loading General Checkpoint for Resuming Training
        ================================================
        To load the items, first initialize the model and optimizer, then load the dictionary 
        locally using torch.load(). From here, you can easily access the saved items by simply 
        querying the dictionary as you would expect.
        '''
        checkpoint = torch.load(path + "model_mario_train.pth", weights_only=True)
        q.load_state_dict(checkpoint['q_state_dict'])
        q_target.load_state_dict(checkpoint['q_target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        n_loss = checkpoint['loss']
        # n_acc = checkpoint['acc']
    
    q.train()
    q_target.train()
    main(epochs, env, q, q_target, optimizer, device)
    
def eval_agent(epochs):
    # ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "mario_q_target.pth"
    # print(f"Load ckpt from {ckpt_path}")
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = model(n_frame, env.action_space.n, device).to(device)
    
    q.load_state_dict(torch.load(path + 'mario_q_target.pth', map_location=torch.device(device), weights_only=True))
    
    total_score = 0.0
    q.eval()
    for k in range(epochs):
        s = arange(env.reset())
        done = False
        score_ = 0.0
        while not done:
            env.render()
            if device == "cpu":
                a = np.argmax(q(s).detach().numpy())
            else:
                a = np.argmax(q(s).cpu().detach().numpy())
            s_prime, r, done, _ = env.step(a)
            s_prime = arange(s_prime)
            score_ += r
            total_score += r
            s = s_prime
            time.sleep(0.01)
    
        stage = env.unwrapped._stage
        print("score : %.1f | total score : %.1f | stage : %d" % (score_, total_score, stage))
        
    env.reset()
    env.close()

def plot_loss(n_loss, n_score):
    plt.figure(figsize = (11,5))
    plt.subplot(1,2,1)
    plt.plot(n_loss)
    plt.xlabel('Épocas')
    plt.ylabel('Função Custo')
    plt.title('loss')
    # plt.xticks()
    # plt.yticks()
    # plt.legend()
    plt.grid()
    plt.subplot(1,2,2)
    plt.plot(n_score)
    plt.xlabel('Épocas')
    plt.ylabel('Score')
    plt.title('Score')
    # plt.xticks()
    # plt.yticks()
    # plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()


# training agent
epochs=3
train_agent(epochs=epochs, resuming_training=True)
a=[]
for i in range(len(loss_lst)):
    a.append(loss_lst[i].detach().cpu().numpy())
a=np.array(a)
plot_loss(a, score_lst)

# eval agent
# eval_agent(7)
# pygame.quit()
