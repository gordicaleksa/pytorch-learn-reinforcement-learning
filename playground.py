import gym
from gym import envs
from stable_baselines3 import PPO
# todo: Explore OpenAI gym capabilities
# todo: set random seeds

# todo: Add DQN
# todo: Add vanilla PG
# todo: Add PPO

# todo: discrete envs: CartPole-v1, Pong, Breakout
# todo: continuous envs: Acrobot, etc.


if __name__ == '__main__':
    # # 1. It renders instance for 500 timesteps, perform random actions
    # env = gym.make('Pong-v4')
    # env.reset()
    # for _ in range(500):
    #     env.render()
    #     observation, reward, done, info = env.step(env.action_space.sample())
    #
    # # 2. To check all env available, uninstalled ones are also shown
    # for el in envs.registry.all():
    #     print(el)

    env = gym.make("CartPole-v1")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
