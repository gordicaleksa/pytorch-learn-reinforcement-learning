import argparse
# todo: what is force used for in Monitor?

import numpy as np
import torch
import matplotlib.pyplot as plt


from utils.utils import get_atari_wrapper
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

    def __init__(self):
        pass

    def learn_from_experience(self):
        pass


class LinearSchedule:

    def __init__(self, schedule_start_value, schedule_end_value, schedule_duration):
        self.start_value = schedule_start_value
        self.end_value = schedule_end_value
        self.schedule_duration = schedule_duration

    def __call__(self, num_steps):
        progress = np.clip(num_steps / self.schedule_duration, a_min=None, a_max=1)
        return self.start_value + (self.end_value - self.start_value) * progress


def train_dqn(config):
    env = get_atari_wrapper(config['env_id'])
    replay_buffer = ReplayBuffer(config['replay_buffer_size'])

    linear_schedule = LinearSchedule(
        config['epsilon_start_value'],
        config['epsilon_end_value'],
        config['epsilon_duration']
    )

    # todo: test the acting part of the pipeline

    # todo: focus on learning part

    # todo: logging, seeds, visualizations
    # todo: train on CartPole verify it's working
    # todo: run on Pong
    # todo: write a readme

    dqn = DQN(env, number_of_actions=env.action_space.n, epsilon_schedule=linear_schedule)
    target_dqn = DQN(env, number_of_actions=env.action_space.n)

    actor = Actor(config, env, replay_buffer, dqn, target_dqn, env.reset())
    learner = Learner()

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
    parser.add_argument("--replay_buffer_size", type=int, help="Number of frames to store in buffer", default=100000)
    parser.add_argument("--acting_learning_ratio", type=int, help="Number of experience steps for every learning step", default=4)
    parser.add_argument("--start_learning", type=int, help="Number of steps before learning starts", default=50000)

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

