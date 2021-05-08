import os
import random


import torch
import git
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gym.wrappers import Monitor


def get_env_wrapper(env_id, record_video=False):
    """
        Ultimately it's not very clear why are SB3's wrappers and OpenAI gym's copy/pasted code for the most part.
        It seems that OpenAI gym doesn't have reward clipping which is necessary for Atari.

        I'm using SB3 because it's actively maintained compared to OpenAI's gym and it has reward clipping by default.

    """
    monitor_dump_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'gym_monitor')

    # This is necessary because AtariWrapper skips 4 frames by default, so we can't have additional skipping through
    # the environment itself - hence NoFrameskip requirement
    assert 'NoFrameskip' in env_id, f'Expected NoFrameskip environment got {env_id}'

    # The only additional thing needed, on top of AtariWrapper,
    # is to convert the shape to channel-first because of PyTorch's models
    env_wrapped = Monitor(ChannelFirst(AtariWrapper(gym.make(env_id))), monitor_dump_dir, force=True, video_callable=lambda episode: record_video)

    return env_wrapped


class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        new_shape = np.roll(self.observation_space.shape, shift=1)  # shape: (H, W, C) -> (C, H, W)

        # Update because this is the last wrapper in the hierarchy, we'll be pooling the env for observation shape info
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)  # shape: (H, W, C) -> (C, H, W)


class LinearSchedule:

    def __init__(self, schedule_start_value, schedule_end_value, schedule_duration):
        self.start_value = schedule_start_value
        self.end_value = schedule_end_value
        self.schedule_duration = schedule_duration

    def __call__(self, num_steps):
        progress = np.clip(num_steps / self.schedule_duration, a_min=None, a_max=1)  # goes from 0 -> 1 and saturates
        return self.start_value + (self.end_value - self.start_value) * progress


class ConstSchedule:
    """ Dummy schedule - used for DQN evaluation in evaluate_dqn_script.py. """
    def __init__(self, value):
        self.value = value

    def __call__(self, num_steps):
        return self.value


def print_model_metadata(training_state):
    header = f'\n{"*"*5} DQN model training metadata: {"*"*5}'
    print(header)

    for key, value in training_state.items():
        if key != 'state_dict':  # don't print state_dict it's a bunch of numbers...
            print(f'{key}: {value}')
    print(f'{"*" * len(header)}\n')


def get_training_state(training_config, model):
    training_state = {
        # Reproducibility details
        "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "seed": training_config['seed'],

        # Env details
        "env_id": training_config['env_id'],

        # Training details
        "best_episode_reward": training_config['best_episode_reward'],

        # Model state
        "state_dict": model.state_dict()
    }

    return training_state


def set_random_seeds(env, seed):
    if seed is not None:
        torch.manual_seed(seed)  # PyTorch
        np.random.seed(seed)  # NumPy
        random.seed(seed)  # Python
        env.action_space.seed(seed)  # probably redundant but I found an article where somebody had a problem with this
        env.seed(seed)  # OpenAI gym

        # todo: AB test impact on FPS metric
        # Deterministic operations for CuDNN, it may impact performances
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Test utils
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    schedule = LinearSchedule(schedule_start_value=1., schedule_end_value=0.1, schedule_duration=50)

    schedule_values = []
    for i in range(100):
        schedule_values.append(schedule(i))

    plt.plot(schedule_values)
    plt.show()

