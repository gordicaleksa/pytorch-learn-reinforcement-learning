import copy


import gym
from gym import envs
from stable_baselines3 import PPO
from torch import nn
import torch
from torchvision import transforms
import numpy as np
import cv2 as cv

# todo: set random seeds

# todo: Add DQN
# todo: Add vanilla PG
# todo: Add PPO

# todo: discrete envs: CartPole-v1, Pong, Breakout
# todo: continuous envs: Acrobot, etc.


ATARI_INPUT = (84, 84)


class ReplayBuffer:
    EMPTY = -1

    def __init__(self):
        self.buffer = []

    def append(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def fetch_random(self):
        random_id = np.random.randint(low=0, high=len(self.buffer)-1)
        old_state, action, reward = self.buffer[random_id]
        new_state, _, _ = self.buffer[random_id+1]
        return old_state, action, reward, new_state

    def fetch_last(self):
        state, _, _ = self.buffer[-1] if len(self.buffer) > 0 else (self.EMPTY, self.EMPTY, self.EMPTY)
        return state

    def update_last(self):
        state = 

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, number_of_actions, input_size):
        super().__init__()
        num_of_neurons_per_layer = [500, 100, 23]
        self.fc1 = nn.Linear(input_size, num_of_neurons_per_layer[0])
        self.fc2 = nn.Linear(num_of_neurons_per_layer[0], num_of_neurons_per_layer[1])
        self.fc3 = nn.Linear(num_of_neurons_per_layer[1], num_of_neurons_per_layer[2])
        self.out = nn.Linear(num_of_neurons_per_layer[2], number_of_actions)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.fc1(input))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.out(x))
        return x


def atari_preprocess(img, current_state, tmp_input_buffer):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_resized = cv.resize(img, (ATARI_INPUT[0], ATARI_INPUT[1]), interpolation=cv.INTER_CUBIC)
    # todo: add handling Atari flicker looking at older image in input_buffer
    return gray_resized


def atari_fetch_input(input_buffer):
    imgs = input_buffer[-4:]
    # todo: convert to PyTorch tensor
    return torch.from_numpy(imgs)


if __name__ == '__main__':
    # # 1. It renders instance for 500 timesteps, perform random actions
    # env = gym.make('Pong-v4')
    # env.reset()
    # for _ in range(500):
    #     env.render()
    #     observation, reward, done, info = env.step(env.action_space.sample())
    #
    # # 2. To check all env available, uninstalled ones are also shown
    # for el in envs.registry.all():
    #     print(el)

    # env = gym.make("CartPole-v1")
    #
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=10000)
    #
    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()
    #
    # env.close()

    env = gym.make("Pong-v4")
    number_of_actions = env.action_space.n
    NUM_EPISODES = 1000
    current_episode = 0
    TARGET_DQN_UPDATE_FREQ = 10

    dqn_current = DQN(number_of_actions=number_of_actions, input_size=ATARI_INPUT[0] * ATARI_INPUT[1])
    dqn_target = DQN(number_of_actions=number_of_actions, input_size=ATARI_INPUT[0] * ATARI_INPUT[1])

    replay_buffer = ReplayBuffer()
    num_sticky_actions = 4

    while current_episode < NUM_EPISODES:
        observation = env.reset()

        end_of_episode = False
        while not end_of_episode:
            current_state = replay_buffer.fetch_last()
            action = dqn_current(current_state) if len(replay_buffer) > 0 else env.action_space.sample()

            tmp_input_buffer = []
            state_reward = 0
            for _ in range(num_sticky_actions):  # sticky actions i.e. repeat the last action num_sticky_actions times
                observation, reward, done, info = env.step(action)
                env.render()

                tmp_input_buffer.append(atari_preprocess(observation, current_state, tmp_input_buffer))
                state_reward += reward

                if done:
                    end_of_episode = True
                    current_episode += 1

            replay_buffer.update_last((action, state_reward))
            replay_buffer.append(tmp_input_buffer)

            # todo: add update function


        if current_episode % TARGET_DQN_UPDATE_FREQ:
            dqn_target = copy.deepcopy(dqn_current)

    env.close()
