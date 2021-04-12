import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper


class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old = self.observation_space.shape
        # todo: why exactly do we need to update observation space?
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(old[-1], old[0], old[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def get_atari_wrapper(env_id):
    assert 'NoFrameskip' in env_id, f'Expected NoFrameskip environment got {env_id}'
    # Add basic Atari processing
    env_wrapped = ChannelFirst(AtariWrapper(gym.make(env_id)))
    return env_wrapped