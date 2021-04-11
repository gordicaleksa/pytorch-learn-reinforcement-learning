from torch import nn
import torch
import numpy as np
import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper


class DQN(nn.Module):
    """
    I wrote it a bit more generic - hence more lines of code -
    but it's more flexible if you want to experiment with the DQN architecture.

    """
    def __init__(self, env, num_in_channels=4, number_of_actions=3):
        super().__init__()
        # CNN params - from the Nature DQN paper
        num_of_filters_cnn = [num_in_channels, 32, 64, 64]
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]

        #
        # Build CNN of DQN
        #
        cnn_modules = []
        for i in range(len(num_of_filters_cnn) - 1):
            cnn_modules.extend(self.cnn_block(num_of_filters_cnn[i], num_of_filters_cnn[i + 1], kernel_sizes[i], strides[i]))

        # todo: make the padding same?
        self.cnn_part = nn.Sequential(
            *cnn_modules,
            nn.Flatten()  # flatten from (B, C, H, W) into (B, C*HxW), where B is batch size and C number of in channels
        )

        #
        # Build fully-connected part of DQN
        #
        with torch.no_grad():  # automatically figure out the shape for the given env observation
            dummy_input = torch.from_numpy(env.observation_space.sample()[np.newaxis])  # shape = (N, 1, H, W)
            if dummy_input.shape[1] != num_in_channels:
                dummy_input = dummy_input.repeat(1, num_in_channels, 1, 1).float()  # convert into (N, C, H, W) float
            num_nodes_fc1 = self.cnn_part(dummy_input).shape[1]  # cnn output shape = (B, C*H*W)

        # FC params
        num_of_neurons_fc = [num_nodes_fc1, 512, number_of_actions]

        fc_modules = []
        for i in range(len(num_of_neurons_fc) - 1):
            last_layer = i == len(num_of_neurons_fc) - 1
            fc_modules.extend(self.fc_block(num_of_neurons_fc[i], num_of_neurons_fc[i + 1], use_relu=not last_layer))

        self.fc_part = nn.Sequential(
            *fc_modules
        )

    def forward(self, observation):
        return self.fc_part(self.cnn_part(observation))

    def cnn_block(self, num_in_filters, num_out_filters, kernel_size, stride):
        layers = [nn.Conv2d(num_in_filters, num_out_filters, kernel_size=kernel_size, stride=stride), nn.ReLU()]
        return layers

    def fc_block(self, num_in_neurons, num_out_neurons, use_relu=True):
        layers = [nn.Linear(num_in_neurons, num_out_neurons)]
        if use_relu:
            layers.append(nn.ReLU())
        return layers


class ChannelFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old = self.observation_space.shape
        # todo: why exactly do we need to update observation space?
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(old[-1], old[0], old[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


# Test DQN network - modular design
if __name__ == '__main__':
    # No frame skipping by OpenAI gym, we receive every frame from the env, and actions are executed deterministically
    env_id = "PongNoFrameskip-v4"
    # Add basic Atari processing
    # todo: can I pass regular Pong?
    env = ChannelFirst(AtariWrapper(gym.make(env_id)))
    dqn = DQN(env)
