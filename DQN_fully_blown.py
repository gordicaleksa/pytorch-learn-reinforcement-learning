import argparse
# todo: what is force used for in Monitor?

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.optim import Adam


from utils.utils import get_atari_wrapper, LinearSchedule
from utils.replay_buffer import ReplayBuffer
from models.definitions.DQN import DQN


class Actor:

    def __init__(self, training_config, env, replay_buffer, dqn, target_dqn, last_frame):
        self.config = training_config
        self.env = env
        self.replay_buffer = replay_buffer
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.last_frame = last_frame
        self.experience_cnt = 0

    def collect_experience(self):
        for _ in range(self.config['acting_learning_ratio']):
            last_index = self.replay_buffer.store_frame(self.last_frame)
            observation = self.replay_buffer.fetch_last_experience()
            # self.visualize_observation(observation)  # <- for debugging
            action = self.sample_action(observation)
            new_frame, reward, done, _ = self.env.step(action)
            self.replay_buffer.store_effect(last_index, action, reward, done)
            if done:
                new_frame = self.env.reset()
            self.last_frame = new_frame
            self.experience_cnt += 1

    def sample_action(self, observation):
        if self.experience_cnt < self.config['start_learning']:
            action = self.env.action_space.sample()  # warm up period
        else:
            with torch.no_grad():
                action = self.dqn.epsilon_greedy(observation)
        return action

    def get_experience_cnt(self):
        return self.experience_cnt

    @staticmethod
    def visualize_observation(observation):
        observation = observation[0].to('cpu').numpy()
        frames = [(img * 255).astype(np.uint8) for img in observation]
        horizontal_img = np.hstack(frames)
        plt.imshow(horizontal_img)
        plt.show()
        plt.close()


class Learner:

    def __init__(self, replay_buffer, dqn, target_dqn, batch_size, gamma, learning_rate):
        # MSE/L2 between [-1,1] and L1 otherwise (as stated in the Nature paper) aka Huber loss
        self.loss = nn.SmoothL1Loss()
        self.replay_buffer = replay_buffer
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.batch_size = batch_size
        self.gamma = gamma
        # todo: using Adam instead of RMSProp, the only difference with Nature paper, btw they haven't specified LR
        # I see some annealing in the original Lua imp
        self.optimizer = Adam(dqn.parameters(), lr=learning_rate)

    # todo: refactor and add shapes info
    def learn_from_experience(self):
        frames_batch, actions_batch, rewards_batch, next_frames_batch, dones_batch = self.replay_buffer.fetch_random_experiences(self.batch_size)

        # Better than detaching: in addition to not being a part of the computational graph it
        # saves time/memory because we're not storing activations during forward propagation needed for the backprop
        with torch.no_grad():
            # shape = (B, 1), [0] because max returns (values, indices) tuples
            next_state_q_values = self.target_dqn(next_frames_batch).max(dim=1, keepdim=True)[0]

            # shape = (B, 1) - implicit broadcasting of rewards and done flags
            target_q_values = rewards_batch + (1 - dones_batch) * self.gamma * next_state_q_values

        current_state_q_values = self.dqn(frames_batch).gather(dim=1, index=actions_batch)

        loss = self.loss(target_q_values, current_state_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # todo: they only mentioned Huber loss in the paper but I see other imps clipping grads
        #  lets log grads and if there is need add clipping
        self.optimizer.step()


def train_dqn(config):
    env = get_atari_wrapper(config['env_id'])
    replay_buffer = ReplayBuffer(config['replay_buffer_size'])

    linear_schedule = LinearSchedule(
        config['epsilon_start_value'],
        config['epsilon_end_value'],
        config['epsilon_duration']
    )

    # todo: test the learning part

    # todo: logging, seeds, visualizations
    # todo: train on CartPole verify it's working

    # todo: run on Pong
    # todo: write a readme

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(env, number_of_actions=env.action_space.n, epsilon_schedule=linear_schedule).to(device)
    target_dqn = DQN(env, number_of_actions=env.action_space.n).to(device)

    actor = Actor(config, env, replay_buffer, dqn, target_dqn, env.reset())
    learner = Learner(replay_buffer, dqn, target_dqn, config['batch_size'], config['gamma'], config['learning_rate'])

    while True:
        if actor.get_experience_cnt() >= config['num_of_training_steps']:
            break

        actor.collect_experience()

        if actor.get_experience_cnt() > config['start_learning']:
            learner.learn_from_experience()

        # todo: logging


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--env_id", type=str, help="environment id for OpenAI gym", default='PongNoFrameskip-v4')
    parser.add_argument("--num_of_training_steps", type=int, help="number of training env steps", default=200000000)
    parser.add_argument("--replay_buffer_size", type=int, help="Number of frames to store in buffer", default=100000) # todo: 1M
    parser.add_argument("--acting_learning_ratio", type=int, help="Number of experience steps for every learning step", default=4)
    parser.add_argument("--start_learning", type=int, help="Number of steps before learning starts", default=100)  # todo: 50000
    parser.add_argument("--batch_size", type=int, help="Number of experiences in a batch (replay buffer)", default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # epsilon-greedy annealing params
    parser.add_argument("--epsilon_start_value", type=float, default=1.)
    parser.add_argument("--epsilon_end_value", type=float, default=0.1)
    parser.add_argument("--epsilon_duration", type=int, default=1000000)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == '__main__':
    # Train the graph attention network (GAT)
    train_dqn(get_training_args())

