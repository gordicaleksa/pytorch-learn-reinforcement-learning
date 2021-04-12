import numpy as np
import psutil


class ReplayBuffer:
    """
    Since stable baselines 3 doesn't currently support a smart replay buffer (more concretely the "lazy frames" feature)
    i.e. allocating (10^6, 84, 84) instead of (10^6, 4, 84, 84) for Atari here is an efficient implementation.

    Note: inspired by Berkley's implementation: https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
    Further improvements:
        * Much more concise (and hopefully readable)
        * Reports error if you don't have enough RAM in advance to allocate this buffer
        * Fixed a subtle bug

    """
    def __init__(self, size, num_last_frames_to_fetch=4, frame_shape=(1, 84, 84), strict=True):
        self.max_buffer_size = size
        self.num_last_frames_to_fetch = num_last_frames_to_fetch

        assert frame_shape[0] == 1 or frame_shape[1] == 3, f'Expected mono/color image frame got shape={frame_shape}.'
        self.frame_height = frame_shape[1]
        self.frame_width = frame_shape[2]

        self.current_buffer_size = 0
        self.current_free_slot_index = 0

        # Create main buffer containers - be aware that numpy does lazy execution so it can happen that after a while
        # you start hitting your RAM limit and your system will start page swapping hence the _check_enough_ram function
        self.frames = np.empty([self.max_buffer_size] + frame_shape, dtype=np.uint8)
        self.actions = np.empty([self.max_buffer_size], dtype=np.uint8)
        self.rewards = np.empty([self.max_buffer_size], dtype=np.float32)
        self.dones = np.empty([self.max_buffer_size], dtype=np.uint8)

        # Basic memory handling since Atari uses 1M frames - and not everybody has a big enough RAM for that
        # todo: write a comment in docs how much memory is needed
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
        self.dones[index] = done  # todo: check the dtype from env

    def fetch_random_experiences(self, batch_size):
        assert self._has_enough_data(batch_size), f"Can't fetch experiences from the replay buffer - not enough data."
        # Uniform random. -1 because we always need to fetch the current and the next successive state for Q-learning,
        # the last state in the buffer doesn't have a successive state
        random_unique_indices = np.random.sample(range(self.current_buffer_size - 1), batch_size)

        frames_batch = np.concatenate([self._fetch_experience(i)[np.newaxis, :] for i in random_unique_indices], 0)
        next_frames_batch = np.concatenate([self._fetch_experience(i + 1)[np.newaxis, :] for i in random_unique_indices], 0)
        actions_batch = self.actions[random_unique_indices]
        rewards_batch = self.rewards[random_unique_indices]
        dones_batch = self.dones[random_unique_indices]

        return frames_batch, actions_batch, rewards_batch, next_frames_batch, dones_batch

    def fetch_last_experience(self):
        return self._fetch_experience((self.current_free_slot_index - 1) % self.max_buffer_size)

    #
    # Helper functions
    #

    def _fetch_experience(self, end_index):
        """
        Replay buffer has 2 edge cases:
            * index is "too close" to 0 and our circular buffer has still not overflown -> not enough frames
            * done flag is True - we don't won't to take frames before that index since it belongs to a different
            life or episode.

        Note: "too close" is defined by 'num_last_frames_to_fetch' variable

        """
        start_index = end_index + 1 - self.num_last_frames_to_fetch

        # todo: open up an issue on berkley's imp there is a subtle bug here
        start_index = self._handle_start_index_edge_case(start_index, end_index)

        
        else:
            # reshape from (PF, C, H, W) to (PF*C, H, W) where PF - number of past frames, usually 4
            return self.frames[start_index:end_index].reshape(-1, self.frame_height, self.frame_width)

    def _handle_start_index_edge_case(self, start_index, end_index):
        c1 = self.current_buffer_size != self.max_buffer_size
        c2 = self.current_buffer_size == self.max_buffer_size and end_index >= self.current_free_slot_index
        if c1:
            start_index = 0
        elif c2:
            start_index = self.current_free_slot_index
        return start_index

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
                      f"Page swapping will make your training super slow once you get to your RAM limit."
            if strict:
                raise Exception(message)
            else:
                print(message)
