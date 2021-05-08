import os


from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN


def run_dqn_baseline():
    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    tensorboard_log = os.path.join(os.path.dirname(__file__), 'runs_baseline')
    buffer_size = 100000
    num_training_steps = 1000000

    model = DQN(
        'CnnPolicy',
        env,
        verbose=0,
        buffer_size=buffer_size,
        learning_starts=50000,
        optimize_memory_usage=False,
        tensorboard_log=tensorboard_log
    )
    model.learn(total_timesteps=num_training_steps)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == '__main__':
    run_dqn_baseline()
