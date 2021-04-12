import numpy as np
import psutil


class ReplayBuffer:
    """
    Since stable baselines 3 doesn't currently support a smart replay buffer (more concretely the "lazy frames" feature)
    i.e. allocating (10^6, 84, 84) instead of (10^6, 4, 84, 84) for Atari here is an efficient implementation.

    Note: inspired by Berkley's implementation: https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
    Further improved:
        * Much more concise (and hopefully readable)
        * Reports error if you don't have enough RAM in advance to allocate this buffer

    """
    def __init__(self, size, num_last_observations_to_fetch=4, frame_shape):
        self.max_buffer_size = size
        self.num_last_observations_to_fetch = num_last_observations_to_fetch

        self.current_buffer_size = 0
        self.current_index = 0

        self.frames = np.empty([self.max_buffer_size] + frame_shape, dtype=np.uint8)
        self.actions = np.empty([self.max_buffer_size], dtype=np.int32)
        self.rewards = np.empty([self.max_buffer_size], dtype=np.float32)
        self.dones = np.empty([self.max_buffer_size], dtype=np.bool)

        mem_available = psutil.virtual_memory().available
        total_memory_usage = self.frames.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
        if total_memory_usage > mem_available:
            raise Exception("Not enough memory to store the complete replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB")

    def store_frame(self, frame):
        self.frames[self.current_index] = frame

        self.current_index = (self.current_index + 1) % self.max_buffer_size  # circular buffer logic
        self.current_buffer_size = min(self.max_buffer_size, self.current_buffer_size + 1)

        return self.current_index - 1  # we yet need to store effect at this index (action, reward, done)

    def store_effect(self, index, action, reward, done):
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done

    def fetch_random_experience(self, batch_size):
        assert self._has_enough_data(batch_size), f"Can't fetch experiences from the replay buffer - not enough data."
        # Uniform random. -1 because we always need to fetch the current and the next successive state for Q-learning,
        # the last state in the buffer doesn't have a successive state
        random_unique_indices = np.random.sample(range(self.current_buffer_size - 1), batch_size)

        obs_batch = np.concatenate([self._encode_observation(i)[np.newaxis, :] for i in random_unique_indices], 0)
        act_batch = self.actions[random_unique_indices]
        rew_batch = self.rewards[random_unique_indices]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[np.newaxis, :] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in random_unique_indices], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch,

    def _has_enough_data(self, batch_size):
        return batch_size < self.current_buffer_size

    def _encode_observation(self, idx):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx - 1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.max_buffer_size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.max_buffer_size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.max_buffer_size])
            return np.concatenate(frames, 0)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.
        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.max_buffer_size)
