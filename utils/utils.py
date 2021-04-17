import os


import git
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gym.wrappers import Monitor


def get_atari_wrapper(env_id):
    """
        Ultimately it's not very clear why are SB3's wrappers and OpenAI gym's copy/pasted code for the most part.
        It seems that OpenAI gym doesn't have reward clipping which is necessary for Atari.

        I'm using SB3 because it's actively maintained compared to OpenAI's gym and it has reward clipping by default.

    """
    # This is necessary because AtariWrapper skips 4 frames by default, so we can't have additional skipping through
    # the environment itself - hence NoFrameskip requirement
    assert 'NoFrameskip' in env_id, f'Expected NoFrameskip environment got {env_id}'

    # The only additional thing needed is to convert the shape to channel-first because of PyTorch's models
    monitor_dump_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'gym_monitor')
    env_wrapped = Monitor(ChannelFirst(AtariWrapper(gym.make(env_id))), monitor_dump_dir, force=True, video_callable=False)

    return env_wrapped


class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        new_shape = np.roll(self.observation_space.shape, shift=1)  # shape: (H, W, C) -> (C, H, W)

        # Update because this is the last wrapper in the hierarchy, we'll be pooling the env for shape info
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


class LinearSchedule:

    def __init__(self, schedule_start_value, schedule_end_value, schedule_duration):
        self.start_value = schedule_start_value
        self.end_value = schedule_end_value
        self.schedule_duration = schedule_duration

    def __call__(self, num_steps):
        progress = np.clip(num_steps / self.schedule_duration, a_min=None, a_max=1)
        return self.start_value + (self.end_value - self.start_value) * progress


def get_training_state(training_config, model):
    training_state = {
        "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,

        # Training details
        "dataset_name": training_config['dataset_name'],
        "num_of_epochs": training_config['num_of_epochs'],
        "test_perf": training_config['test_perf'],

        # Model structure
        "num_of_layers": training_config['num_of_layers'],
        "num_heads_per_layer": training_config['num_heads_per_layer'],
        "num_features_per_layer": training_config['num_features_per_layer'],
        "add_skip_connection": training_config['add_skip_connection'],
        "bias": training_config['bias'],
        "dropout": training_config['dropout'],
        "layer_type": training_config['layer_type'].name,

        # Model state
        "state_dict": model.state_dict()
    }

    return training_state


def set_random_seeds(env, seed):
    import torch
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)  # probably redundant but I found an article where somebody had a problem with this
    env.seed(seed)


# Test utils
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    schedule = LinearSchedule(1., 0.1, 50)

    schedule_values = []
    for i in range(100):
        schedule_values.append(schedule(i))

    plt.plot(schedule_values)
    plt.show()

