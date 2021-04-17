import os


import git
import gym
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper, WarpFrame
from gym.wrappers import Monitor


# todo: maybe try out CartPole but via image input
# def cart_pole():
#     resize = T.Compose([T.ToPILImage(),
#                         T.Scale(40, interpolation=Image.CUBIC),
#                         T.ToTensor()])
#
#     # This is based on the code from gym.
#     screen_width = 600
#
#     def get_cart_location():
#         world_width = env.x_threshold * 2
#         scale = screen_width / world_width
#         return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
#
#     def get_screen():
#         screen = env.render(mode='rgb_array').transpose(
#             (2, 0, 1))  # transpose into torch order (CHW)
#         # Strip off the top and bottom of the screen
#         screen = screen[:, 160:320]
#         view_width = 320
#         cart_location = get_cart_location()
#         if cart_location < view_width // 2:
#             slice_range = slice(view_width)
#         elif cart_location > (screen_width - view_width // 2):
#             slice_range = slice(-view_width, None)
#         else:
#             slice_range = slice(cart_location - view_width // 2,
#                                 cart_location + view_width // 2)
#         # Strip off the edges, so that we have a square image centered on a cart
#         screen = screen[:, :, slice_range]
#         # Convert to float, rescare, convert to torch tensor
#         # (this doesn't require a copy)
#         screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
#         screen = torch.from_numpy(screen)
#         # Resize, and add a batch dimension (BCHW)
#         return resize(screen).unsqueeze(0).type(Tensor)


def get_env_wrapper(env_id):
    """
        Ultimately it's not very clear why are SB3's wrappers and OpenAI gym's copy/pasted code for the most part.
        It seems that OpenAI gym doesn't have reward clipping which is necessary for Atari.

        I'm using SB3 because it's actively maintained compared to OpenAI's gym and it has reward clipping by default.

    """
    monitor_dump_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'gym_monitor')

    if env_id == 'CartPole-v1':
        env_wrapped = Monitor(gym.make(env_id), monitor_dump_dir, force=True, video_callable=False)
    else:
        # This is necessary because AtariWrapper skips 4 frames by default, so we can't have additional skipping through
        # the environment itself - hence NoFrameskip requirement
        assert 'NoFrameskip' in env_id, f'Expected NoFrameskip environment got {env_id}'

        # The only additional thing needed is to convert the shape to channel-first because of PyTorch's models
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
    import torch
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # todo: AB test whether I get the same results with/without this line
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

