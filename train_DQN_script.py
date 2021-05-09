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
import copy


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


class ActorLearner:

    def __init__(self, config, env, replay_buffer, dqn, target_dqn, last_frame):

        self.start_time = time.time()

        self.config = config
        self.env = env
        self.last_frame = last_frame  # always keeps the latest frame from the environment
        self.replay_buffer = replay_buffer

        # DQN Models
        self.dqn = dqn
        self.target_dqn = target_dqn

        # Logging/debugging-related
        self.debug = config['debug']
        self.log_freq = config['log_freq']
        self.episode_log_freq = config['episode_log_freq']
        self.grads_log_freq = config['grads_log_freq']
        self.checkpoint_freq = config['checkpoint_freq']
        self.tensorboard_writer = SummaryWriter()
        self.huber_loss = []
        self.best_episode_reward = -np.inf
        self.best_dqn_model = None  # keeps a deep copy of the best DQN model so far (best = highest episode reward)

        # MSE/L2 between [-1,1] and L1 otherwise (as stated in the Nature paper) aka "Huber loss"
        self.loss = nn.SmoothL1Loss()
        self.optimizer = Adam(self.dqn.parameters(), lr=config['learning_rate'])
        self.grad_clip_value = config['grad_clipping_value']

        self.acting_learning_step_ratio = config['acting_learning_step_ratio']
        self.num_warmup_steps = config['num_warmup_steps']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']  # discount factor

        self.learner_cnt = 0
        self.target_dqn_update_interval = config['target_dqn_update_interval']
        # should perform a hard or a soft update of target DQN weights
        self.tau = config['tau']

    def collect_experience(self):
        # We're collecting more experience than we're doing weight updates (4x in the Nature paper)
        for _ in range(self.acting_learning_step_ratio):
            last_index = self.replay_buffer.store_frame(self.last_frame)
            state = self.replay_buffer.fetch_last_state()  # state = 4 preprocessed last frames for Atari

            action = self.sample_action(state)
            new_frame, reward, done_flag, _ = self.env.step(action)

            self.replay_buffer.store_action_reward_done(last_index, action, reward, done_flag)

            if done_flag:
                new_frame = self.env.reset()
                self.maybe_log_episode()

            self.last_frame = new_frame

            if self.debug:
                self.visualize_state(state)
                self.env.render()

            self.maybe_log()

    def sample_action(self, state):
        if self.env.get_total_steps() < self.num_warmup_steps:
            action = self.env.action_space.sample()  # initial warm up period - no learning, acting randomly
        else:
            with torch.no_grad():
                action = self.dqn.epsilon_greedy(state)
        return action

    def get_number_of_env_steps(self):
        return self.env.get_total_steps()

    def learn_from_experience(self):
        current_states, actions, rewards, next_states, done_flags = self.replay_buffer.fetch_random_states(self.batch_size)

        # Better than detaching: in addition to target dqn not being a part of the computational graph it also
        # saves time/memory because we're not storing activations during forward propagation needed for the backprop
        with torch.no_grad():
            # shape = (B, NA) -> (B, 1), where NA - number of actions
            # [0] because max returns (values, indices) tuples
            next_state_max_q_values = self.target_dqn(next_states).max(dim=1, keepdim=True)[0]

            # shape = (B, 1), TD targets. We need (1 - done) because when we're in a terminal state the next
            # state Q value should be 0 and we only use the reward information
            target_q_values = rewards + (1 - done_flags) * self.gamma * next_state_max_q_values

        # shape = (B, 1), pick those Q values that correspond to the actions we made in those states
        current_state_q_values = self.dqn(current_states).gather(dim=1, index=actions)

        loss = self.loss(target_q_values, current_state_q_values)
        self.huber_loss.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()  # compute the gradients

        if self.grad_clip_value is not None:  # potentially clip gradients for stability reasons
            nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip_value)

        self.optimizer.step()  # update step
        self.learner_cnt += 1

        # Periodically update the target DQN weights (coupled to the number of DQN weight updates and not # env steps)
        if self.learner_cnt % self.target_dqn_update_interval == 0:
            if self.tau == 1.:
                print('Update target DQN (hard update)')
                self.target_dqn.load_state_dict(self.dqn.state_dict())
            else:  # soft update, the 2 branches can be merged together, leaving it like this for now
                raise Exception(f'Soft update is not yet implemented (hard update was used in the original paper)')

    @staticmethod
    def visualize_state(state):
        state = state[0].to('cpu').numpy()  # (1/B, C, H, W) -> (C, H, W)
        stacked_frames = np.hstack([np.repeat((img * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2) for img in state])  # (C, H, W) -> (H, C*W, 3)
        plt.imshow(stacked_frames)
        plt.show()

    def maybe_log_episode(self):
        rewards = self.env.get_episode_rewards()  # we can do this thanks to the Monitor wrapper
        episode_lengths = self.env.get_episode_lengths()
        num_episodes = len(rewards)

        if self.episode_log_freq is not None and num_episodes % self.episode_log_freq == 0:
            self.tensorboard_writer.add_scalar('Rewards per episode', rewards[-1], num_episodes)
            self.tensorboard_writer.add_scalar('Steps per episode', episode_lengths[-1], num_episodes)

        if rewards[-1] > self.best_episode_reward:
            self.best_episode_reward = rewards[-1]
            self.config['best_episode_reward'] = self.best_episode_reward  # metadata
            self.best_dqn_model = copy.deepcopy(self.dqn)  # keep track of the model that gave the best reward

    def maybe_log(self):
        num_steps = self.env.get_total_steps()

        if self.log_freq is not None and num_steps > 0 and num_steps % self.log_freq == 0:
            self.tensorboard_writer.add_scalar('Epsilon', self.dqn.epsilon_value(), num_steps)
            if len(self.huber_loss) > 0:
                self.tensorboard_writer.add_scalar('Huber loss', np.mean(self.huber_loss), num_steps)
            self.tensorboard_writer.add_scalar('FPS', num_steps / (time.time() - self.start_time), num_steps)

            self.huber_loss = []  # clear the loss values and start recollecting them again

        # Periodically save DQN models
        if self.checkpoint_freq is not None and num_steps > 0 and num_steps % self.checkpoint_freq == 0:
            ckpt_model_name = f'dqn_{self.config["env_id"]}_ckpt_steps_{num_steps}.pth'
            torch.save(utils.get_training_state(self.config, self.dqn), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

        # Log the gradients
        if self.grads_log_freq is not None and self.learner_cnt > 0 and self.learner_cnt % self.grads_log_freq == 0:
            total_grad_l2_norm = 0

            for cnt, (name, weight_or_bias_parameters) in enumerate(self.dqn.named_parameters()):
                grad_l2_norm = weight_or_bias_parameters.grad.data.norm(p=2).item()
                self.tensorboard_writer.add_scalar(f'grad_norms/{name}', grad_l2_norm, self.learner_cnt)
                total_grad_l2_norm += grad_l2_norm ** 2

            # As if we concatenated all of the params into a single vector and took L2
            total_grad_l2_norm = total_grad_l2_norm ** (1/2)
            self.tensorboard_writer.add_scalar(f'grad_norms/total', total_grad_l2_norm, self.learner_cnt)

    def log_to_console(self):  # keep it minimal for now, I mostly use tensorboard - feel free to expand functionality
        print(f'Number of env steps = {self.get_number_of_env_steps()}')


def train_dqn(config):
    env = utils.get_env_wrapper(config['env_id'])
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], crash_if_no_mem=config['dont_crash_if_no_mem'])

    utils.set_random_seeds(env, config['seed'])

    linear_schedule = utils.LinearSchedule(
        config['epsilon_start_value'],
        config['epsilon_end_value'],
        config['epsilon_duration']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(env, number_of_actions=env.action_space.n, epsilon_schedule=linear_schedule).to(device)
    target_dqn = DQN(env, number_of_actions=env.action_space.n).to(device)

    # Don't get confused by the actor-learner terminology, DQN is not an actor-critic method, but conceptually
    # we can split the learning process into collecting experience/acting in the env and learning from that experience
    actor_learner = ActorLearner(config, env, replay_buffer, dqn, target_dqn, env.reset())

    while actor_learner.get_number_of_env_steps() < config['num_of_training_steps']:

        num_env_steps = actor_learner.get_number_of_env_steps()
        if config['console_log_freq'] is not None and num_env_steps % config['console_log_freq'] == 0:
            actor_learner.log_to_console()

        actor_learner.collect_experience()

        if num_env_steps > config['num_warmup_steps']:
            actor_learner.learn_from_experience()

    torch.save(  # save the best DQN model overall (gave the highest reward in an episode)
        utils.get_training_state(config, actor_learner.best_dqn_model),
        os.path.join(BINARIES_PATH, utils.get_available_binary_name(config['env_id']))
    )


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--seed", type=int, help="Very important for reproducibility - set the random seed", default=23)
    parser.add_argument("--env_id", type=str, help="Atari game id", default='BreakoutNoFrameskip-v4')
    parser.add_argument("--num_of_training_steps", type=int, help="Number of training env steps", default=50000000)
    parser.add_argument("--acting_learning_step_ratio", type=int, help="Number of experience collection steps for every learning step", default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--grad_clipping_value", type=float, default=5)  # 5 is fairly arbitrarily chosen

    parser.add_argument("--replay_buffer_size", type=int, help="Number of frames to store in buffer", default=1000000)
    parser.add_argument("--dont_crash_if_no_mem", action='store_false', help="Optimization - crash if not enough RAM before the training even starts (default=True)")
    parser.add_argument("--num_warmup_steps", type=int, help="Number of steps before learning starts", default=50000)
    parser.add_argument("--target_dqn_update_interval", type=int, help="Target DQN update freq per learning update", default=10000)

    parser.add_argument("--batch_size", type=int, help="Number of states in a batch (from replay buffer)", default=32)
    parser.add_argument("--gamma", type=float, help="Discount factor", default=0.99)
    parser.add_argument("--tau", type=float, help='Set to 1 for a hard target DQN update, < 1 for a soft one', default=1.)

    # epsilon-greedy annealing params
    parser.add_argument("--epsilon_start_value", type=float, default=1.)
    parser.add_argument("--epsilon_end_value", type=float, default=0.1)
    parser.add_argument("--epsilon_duration", type=int, default=1000000)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--console_log_freq", type=int, help="Log to console after this many env steps (None = no logging)", default=10000)
    parser.add_argument("--log_freq", type=int, help="Log metrics to tensorboard after this many env steps (None = no logging)", default=10000)
    parser.add_argument("--episode_log_freq", type=int, help="Log metrics to tensorboard after this many episodes (None = no logging)", default=5)
    parser.add_argument("--checkpoint_freq", type=int, help="Save checkpoint model after this many env steps (None = no checkpointing)", default=10000)
    parser.add_argument("--grads_log_freq", type=int, help="Log grad norms after this many weight update steps (None = no logging)", default=2500)
    parser.add_argument("--debug", action='store_true', help='Train in debugging mode')
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == '__main__':
    # Train the DQN model
    train_dqn(get_training_args())


