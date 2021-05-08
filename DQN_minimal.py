from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = DQN('CnnPolicy', env, verbose=1, buffer_size=100000, learning_starts=100, optimize_memory_usage=True)
model.learn(total_timesteps=1000000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()