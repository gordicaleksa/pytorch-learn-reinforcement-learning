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


class LinearSchedule:

    def __init__(self, schedule_start_value, schedule_end_value, schedule_duration):
        self.start_value = schedule_start_value
        self.end_value = schedule_end_value
        self.schedule_duration = schedule_duration

    def __call__(self, num_steps):
        progress = np.clip(num_steps / self.schedule_duration, a_min=None, a_max=1)
        return self.start_value + (self.end_value - self.start_value) * progress


# Test utils
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    schedule = LinearSchedule(1., 0.1, 50)
    schedule_values = []
    for i in range(100):
        schedule_values.append(schedule(i))

    plt.plot(schedule_values)
    plt.show()
