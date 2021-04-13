import argparse
from itertools import count
# todo: what is force used for in Monitor?


from utils.utils import get_atari_wrapper
from utils.replay_buffer import ReplayBuffer
from models.definitions.DQN import DQN


class ExperienceCollector:

    def __init__(self, training_config, env, replay_buffer, dqn, target_dqn, last_frame):
        self.config = training_config
        self.env = env
        self.replay_buffer = replay_buffer
        self.dqn = dqn
        self.target_dqn = target_dqn
        self.last_frame = last_frame

    def collect(self, experience_cnt):
        for _ in range(self.config[''])
            last_index = self.replay_buffer.store_frame(self.last_frame)
            observation = self.replay_buffer.fetch_last_experience()
            action = self.sample_action(observation, experience_cnt)
            new_observation, reward, done, _ = self.env.step(action)
            self.replay_buffer.store_effect(last_index, action, reward, done)
            if done:
                obs = self.env.reset()
            last_obs = obs

    def sample_action(self, observation, experience_cnt):
        if experience_cnt < self.config['start_learning']:
            action = self.env.action_space.sample()  # warm up period
        else:
            action = self.dqn(observation)
        return action


def train_dqn(training_config):
    env = get_atari_wrapper(training_config['env_id'])
    replay_buffer = ReplayBuffer(training_config['replay_buffer_size'])

    dqn = DQN(env, number_of_actions=env.action_space.n)
    target_dqn = DQN(env, number_of_actions=env.action_space.n)

    experience_collector = ExperienceCollector(training_config, env, replay_buffer, dqn, target_dqn, env.reset())

    experience_cnt = 0
    while True:
        if experience_cnt >= training_config['num_of_training_steps']:
            break

        experience_cnt += experience_collector.collect(experience_cnt)

        if experience_cnt > training_config['start_learning']:
            dqn_learn_from_experience()

        # log


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--env_id", type=str, help="environment id for OpenAI gym", default='PongNoFrameskip-v4')
    parser.add_argument("--num_of_training_steps", type=int, help="number of training env steps", default=200000000)
    parser.add_argument("--replay_buffer_size", type=int, help="Number of frames to store in buffer", default=1000000)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == '__main__':
    # Train the graph attention network (GAT)
    train_dqn(get_training_args())

