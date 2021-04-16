import random


import numpy as np
import psutil
import torch


from utils.utils import get_atari_wrapper


# todo: decide on naming - observations or experience
class ReplayBuffer:
    """
    Since stable baselines 3 doesn't currently support a smart replay buffer (more concretely the "lazy frames" feature)
    i.e. allocating (10^6, 84, 84) (~7 GB) instead of (10^6, 4, 84, 84) for Atari here is an efficient implementation.

    Note: inspired by Berkley's implementation: https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3

    Further improvements:
        * Much more concise (and hopefully readable)
        * Reports error if you don't have enough RAM in advance to allocate this buffer
        * Fixed a subtle buffer boundary bug (start index edge case)

    """
    def __init__(self, max_size, num_last_frames_to_fetch=4, frame_shape=[1, 84, 84], strict=True):
        self.max_buffer_size = max_size
        self.num_previous_frames_to_fetch = num_last_frames_to_fetch

        assert frame_shape[0] == 1 or frame_shape[1] == 3, f'Expected mono/color image frame got shape={frame_shape}.'
        self.frame_height = frame_shape[1]
        self.frame_width = frame_shape[2]

        self.current_buffer_size = 0
        self.current_free_slot_index = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create main buffer containers - be aware that numpy does lazy execution so it can happen that after a while
        # you start hitting your RAM limit and your system will start page swapping hence the _check_enough_ram function
        self.frames = np.empty([self.max_buffer_size] + frame_shape, dtype=np.uint8)
        self.actions = np.empty([self.max_buffer_size, 1], dtype=np.uint8)
        self.rewards = np.empty([self.max_buffer_size, 1], dtype=np.float32)
        self.dones = np.empty([self.max_buffer_size, 1], dtype=np.uint8)

        # Basic memory handling since Atari uses 1M frames - and not everybody has a big enough RAM for that
        self._check_enough_ram(strict)

    #
    # public API functions
    #

    def store_frame(self, frame):
        self.frames[self.current_free_slot_index] = frame

        self.current_free_slot_index = (self.current_free_slot_index + 1) % self.max_buffer_size  # circular buffer logic
        self.current_buffer_size = min(self.max_buffer_size, self.current_buffer_size + 1)

        return self.current_free_slot_index - 1  # we yet need to store effect at this index (action, reward, done)

    def store_effect(self, index, action, reward, done):
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

    def fetch_random_experiences(self, batch_size):
        assert self._has_enough_data(batch_size), f"Can't fetch experiences from the replay buffer - not enough data."
        # Uniform random. -1 because we always need to fetch the current and the next successive state for Q-learning,
        # the last state in the buffer doesn't have a successive state
        random_unique_indices = random.sample(range(self.current_buffer_size - 1), batch_size)

        frames_batch = self._postprocess_experience(np.concatenate([self._fetch_experience(i)[np.newaxis, :] for i in random_unique_indices], 0))
        next_frames_batch = self._postprocess_experience(np.concatenate([self._fetch_experience(i + 1)[np.newaxis, :] for i in random_unique_indices], 0))
        actions_batch = torch.from_numpy(self.actions[random_unique_indices]).to(self.device).long()
        rewards_batch = torch.from_numpy(self.rewards[random_unique_indices]).to(self.device)
        dones_batch = torch.from_numpy(self.dones[random_unique_indices]).to(self.device).float()

        return frames_batch, actions_batch, rewards_batch, next_frames_batch, dones_batch

    def fetch_last_experience(self):
        return self._postprocess_experience(self._fetch_experience((self.current_free_slot_index - 1) % self.max_buffer_size))

    #
    # Helper functions
    #

    def _fetch_experience(self, end_index):
        """
        We fetch end_index frame and ("num_last_frames_to_fetch" - 1) last frames (4 in total in the case of Atari).

        Replay buffer has 2 edge cases:
            * index is "too close" to 0 and our circular buffer has still not overflown -> not enough frames
            * done flag is True - we don't won't to take frames before that index since it belongs to a different
            life or episode.

        Note: "too close" is defined by 'num_last_frames_to_fetch' variable

        """
        # Start/end indices are inclusive [] and not [),(],()
        start_index = end_index + 1 - self.num_previous_frames_to_fetch

        start_index = self._handle_start_index_edge_cases(start_index)
        start_index = self._handle_done_flag(start_index, end_index)

        num_of_missing_frames = self.num_previous_frames_to_fetch - (end_index + 1 - start_index)

        if start_index < 0 or num_of_missing_frames > 0:  # start_index:end_index won't work if start_index < 0
            # If there are missing frames, because of the above handled edge-cases, fill them with black frames as per
            # original DeepMind Lua imp: https://github.com/deepmind/dqn/blob/master/dqn/TransitionTable.lua#L171
            experience = [np.zeros_like(self.frames[0]) for _ in range(num_of_missing_frames)]

            for index in range(start_index, end_index + 1):
                experience.append(self.frames[index % self.max_buffer_size])

            # shape = (PF*C, H, W) where PF - number of Past Frames, 4 for Atari
            return np.concatenate(experience, 0)
        else:
            # reshape from (PF, C, H, W) to (PF*C, H, W) where PF - number of Past Frames, 4 for Atari
            return self.frames[start_index:end_index + 1].reshape(-1, self.frame_height, self.frame_width)

    def _postprocess_experience(self, observation):
        if observation.ndim == 3:
            observation = np.expand_dims(observation, 0)  # shape: (PF*C, H, W) -> (1, PF*C, H, W)
        # numpy -> tensor, move to device, uint8 -> float, [0,255] -> [0, 1]
        return torch.from_numpy(observation).to(self.device).float().div(255)

    def _handle_start_index_edge_cases(self, start_index):
        if not self._buffer_full() and start_index < 0:
            start_index = 0
        # Handle the case where start index crosses the buffer head pointer - the data before and after the head pointer
        # belongs to completely different episodes
        if self._buffer_full():
            if 0 < self.current_free_slot_index - start_index < self.num_previous_frames_to_fetch:
                start_index = self.current_free_slot_index
        return start_index

    def _handle_done_flag(self, start_index, end_index):
        new_start_index = start_index

        # A done flag marks a boundary between different episodes or lives either way we shouldn't take frames
        # before or at the done flag into consideration
        for index in range(start_index, end_index):  # no + 1 here since done flag is not yet set for end_index + 1
            if self.dones[index % self.max_buffer_size]:
                new_start_index = index + 1

        return new_start_index

    def _buffer_full(self):
        return self.current_buffer_size == self.max_buffer_size

    def _has_enough_data(self, batch_size):
        return batch_size < self.current_buffer_size

    def _check_enough_ram(self, strict):
        def to_GBs(memory_in_bytes):  # beautify memory output - helper function
            return f'{memory_in_bytes / 2 ** 30:.2f} GBs'

        available_memory = psutil.virtual_memory().available
        total_memory = self.frames.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

        if total_memory > available_memory:
            message = f"Not enough memory to store the complete replay buffer! \n" \
                      f"total:{to_GBs(total_memory)} > available:{to_GBs(available_memory)} \n" \
                      f"Page swapping will make your training super slow once you hit your RAM limit."
            if strict:
                raise Exception(message)
            else:
                print(message)


# Basic replay buffer testing
if __name__ == '__main__':
    size = 500000
    num_of_collection_steps = 10000
    experience_batch_size = 32

    # Step 0: Create replay buffer and the env
    replay_buffer = ReplayBuffer(size)

    # NoFrameskip - receive every frame from the env whereas the version without NoFrameskip would give every 4th frame
    # v4 - actions we send to env are executed, whereas v0 would execute the last action we sent with 0.25 probability
    env_id = "PongNoFrameskip-v4"
    env = get_atari_wrapper(env_id)

    # Step 1: Add some experience to the buffer
    frame = env.reset()

    for i in range(num_of_collection_steps):
        random_action = env.action_space.sample()
        print(f'Sampling action {random_action} - {env.unwrapped.get_action_meanings()[random_action]}')
        frame, reward, done, info = env.step(random_action)
        index = replay_buffer.store_frame(frame)
        replay_buffer.store_effect(index, random_action, reward, done)
        if done:
            env.reset()

    # Step 2: Fetch experiences from the buffer
    frames_batch, actions_batch, rewards_batch, next_frames_batch, dones_batch = replay_buffer.fetch_random_experiences(experience_batch_size)

    print(frames_batch.shape, next_frames_batch.shape, actions_batch.shape, rewards_batch.shape, dones_batch.shape)
