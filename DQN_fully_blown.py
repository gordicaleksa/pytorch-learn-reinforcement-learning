"""
    Implementation of the original DQN Nature paper:
        https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

    Some of the complexity is captured via wrappers but the main components such as the DQN model itself,
    the training loop, the memory-efficient replay buffer are implemented from scratch.

    Some modifications:
        * Using Adam instead of RMSProp

"""

import os
import argparse
import time


import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


import utils.utils as utils
from utils.replay_buffer import ReplayBuffer
from utils.constants import *
from models.definitions.DQN import DQN


# todo: train on CartPole verify it's working (TEST WITH SMALL REPLAY BUFFER TO CATCH BUGS)

# todo: visualizations (dump episode video from time to time using monitor)
# todo: run on Pong
# todo: write a readme
class ActorLearner:

    def __init__(self, config, env, replay_buffer, dqn, target_dqn, last_frame):
        #
        # Actor params
        #

        self.start_time = time.time()
        self.config = config
        self.env = env
        self.replay_buffer = replay_buffer
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.last_frame = last_frame  # always keeps the latest frame from the environment

        self.log_freq = config['log_freq']
        self.episode_log_freq = config['episode_log_freq']
        self.checkpoint_freq = config['checkpoint_freq']
        self.tensorboard_writer = SummaryWriter()

        #
        # Learner params
        #

        # MSE/L2 between [-1,1] and L1 otherwise (as stated in the Nature paper) aka "Huber loss"
        self.acting_learning_ratio = config['acting_learning_ratio']
        self.start_learning = config['start_learning']
        self.loss = nn.SmoothL1Loss()
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        # todo: experiment with RMSProp, the only difference with Nature paper, btw they haven't specified LR
        # todo: I see some LR annealing in the original Lua imp
        # todo: read Adam and RMSProp papers
        self.optimizer = Adam(dqn.parameters(), lr=config['learning_rate'])
        self.learner_cnt = 0
        self.target_dqn_update_interval = config['target_dqn_update_interval']
        # should perform a hard or a soft update of target DQN weights
        self.hard_target_update = config['is_hard_target_update']

        self.huber_loss = []
        self.best_episode_reward = -np.inf

    def collect_experience(self):
        # We're collecting more experience than we're doing weight updates (4x in the Nature paper)
        for _ in range(self.acting_learning_ratio):
            last_index = self.replay_buffer.store_frame(self.last_frame)
            observation = self.replay_buffer.fetch_last_observation()
            # self.visualize_observation(observation)  # <- for debugging
            action = self.sample_action(observation)
            new_frame, reward, done, _ = self.env.step(action)
            self.replay_buffer.store_effect(last_index, action, reward, done)
            if done:
                new_frame = self.env.reset()
                self.maybe_log_episode()

            self.last_frame = new_frame

            self.maybe_log()

    def sample_action(self, observation):
        if self.env.get_total_steps() < self.start_learning:
            action = self.env.action_space.sample()  # initial warm up period - no learning and acting randomly
        else:
            with torch.no_grad():
                action = self.dqn.epsilon_greedy(observation)
        return action

    def get_experience_cnt(self):
        return self.env.get_total_steps()

    def learn_from_experience(self):
        observations, actions, rewards, next_observations, dones = self.replay_buffer.fetch_random_observations(self.batch_size)

        # Better than detaching: in addition to target dqn not being a part of the computational graph it also
        # saves time/memory because we're not storing activations during forward propagation needed for the backprop
        with torch.no_grad():
            # shape = (B, 1), [0] because max returns (values, indices) tuples
            next_state_q_values = self.target_dqn(next_observations).max(dim=1, keepdim=True)[0]

            # shape = (B, 1), forming TD targets, we need (1 - done) because when we're in a terminal state the next
            # state Q value should be 0 and we only use the reward information
            target_q_values = rewards + (1 - dones) * self.gamma * next_state_q_values

        # shape = (B, 1), pick those Q values that correspond to the actions we did in that point of time
        current_state_q_values = self.dqn(observations).gather(dim=1, index=actions)

        loss = self.loss(target_q_values, current_state_q_values)
        self.huber_loss.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        # todo: they only mentioned Huber loss in the paper but I see other imps clipping grads
        #  lets log grads and if there is need add clipping
        self.optimizer.step()
        self.learner_cnt += 1

        # Periodically update the target DQN weights, coupled to weight updates and not env steps
        if self.learner_cnt % self.target_dqn_update_interval == 0:
            if self.hard_target_update:
                self.target_dqn.load_state_dict(self.dqn.state_dict())
            else:  # soft update
                raise Exception(f'Not yet implemented')

    @staticmethod
    def visualize_observation(observation):
        observation = observation[0].to('cpu').numpy()  # (1, C, H, W) -> (C, H, W)
        stacked_frames = np.hstack([(img * 255).astype(np.uint8) for img in observation])  # (C, H, W) -> (H, C*W)
        plt.imshow(stacked_frames)
        plt.show()
        plt.close()

    def maybe_log_episode(self):
        rewards = self.env.get_episode_rewards()
        episode_lengths = self.env.get_episode_lengths()
        num_episodes = len(rewards)
        if self.episode_log_freq is not None and num_episodes % self.episode_log_freq == 0:
            self.tensorboard_writer.add_scalar('Reward per episode', rewards[-1], num_episodes)
            self.tensorboard_writer.add_scalar('Steps per episode', episode_lengths[-1], num_episodes)

            self.best_episode_reward = max(self.best_episode_reward, rewards[-1])
            self.config['best_episode_reward'] = self.best_episode_reward

    def maybe_log(self):
        num_steps = self.env.get_total_steps()

        if self.log_freq is not None and num_steps % self.log_freq == 0:
            self.tensorboard_writer.add_scalar('Epsilon', self.dqn.epsilon_value(), num_steps)
            self.tensorboard_writer.add_scalar('Huber loss', np.mean(self.huber_loss), num_steps)
            self.tensorboard_writer.add_scalar('FPS', num_steps / (time.time() - self.start_time), num_steps)

            self.huber_loss = []

        if self.checkpoint_freq is not None and num_steps % self.checkpoint_freq == 0:
            ckpt_model_name = f'dqn_{self.config["env_id"]}_ckpt_steps_{num_steps}.pth'
            torch.save(utils.get_training_state(self.config, self.dqn), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))


def train_dqn(config):
    env = utils.get_env_wrapper(config['env_id'])
    replay_buffer = ReplayBuffer(config['replay_buffer_size'])
    utils.set_random_seeds(env, config['seed'])

    linear_schedule = utils.LinearSchedule(
        config['epsilon_start_value'],
        config['epsilon_end_value'],
        config['epsilon_duration']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(env, number_of_actions=env.action_space.n, epsilon_schedule=linear_schedule).to(device)
    target_dqn = DQN(env, number_of_actions=env.action_space.n).to(device)

    # Don't get confused by the actor-learner terminology, DQN is not actor-critic method, but conceptually
    # we can split the learning process into collecting experience/acting in the env and learning from that experience
    actor_learner = ActorLearner(config, env, replay_buffer, dqn, target_dqn, env.reset())

    while actor_learner.get_experience_cnt() < config['num_of_training_steps']:

        actor_learner.collect_experience()

        if actor_learner.get_experience_cnt() > config['start_learning']:
            actor_learner.learn_from_experience()


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--seed", type=int, help="Very important for reproducibility - set random seed", default=42)
    parser.add_argument("--env_id", type=str, help="environment id for OpenAI gym", default='PongNoFrameskip-v4')  # todo: CartPole-v1
    parser.add_argument("--num_of_training_steps", type=int, help="Number of training env steps", default=5000000)  # todo: 50000000
    parser.add_argument("--replay_buffer_size", type=int, help="Number of frames to store in buffer", default=400000) # todo: 1M
    parser.add_argument("--acting_learning_ratio", type=int, help="Number of experience steps for every learning step", default=4)
    parser.add_argument("--start_learning", type=int, help="Number of steps before learning starts", default=10000)  # todo: 50000
    parser.add_argument("--target_dqn_update_interval", type=int, help="Target DQN update freq per learning update", default=10000)
    parser.add_argument("--batch_size", type=int, help="Number of experiences in a batch (replay buffer)", default=32)
    parser.add_argument("--gamma", type=float, help="Discount factor", default=0.99)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--is_hard_target_update", action='store_true', help='perform hard target DQN update')

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--log_freq", type=int, help="Log various metrics to tensorboard after this many env steps (None no logging)", default=1000)
    parser.add_argument("--episode_log_freq", type=int, help="Log various metrics to tensorboard after this many episodes (None no logging)", default=1)
    parser.add_argument("--checkpoint_freq", type=int, help="Save checkpoint model after this many env steps (None for no checkpointing)", default=10000)

    # epsilon-greedy annealing params
    parser.add_argument("--epsilon_start_value", type=float, default=1.)
    parser.add_argument("--epsilon_end_value", type=float, default=0.1)
    parser.add_argument("--epsilon_duration", type=int, default=50000)  # todo: 1000000
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == '__main__':
    # Train the DQN model
    train_dqn(get_training_args())

