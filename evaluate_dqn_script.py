import os


import torch


import utils.utils as utils
from models.definitions.DQN import DQN
from utils.constants import *
from utils.replay_buffer import ReplayBuffer


if __name__ == '__main__':
    buffer_size = 100000
    epsilon_eval = 0.05
    env_id = 'BreakoutNoFrameskip-v4'
    model_name = 'dqn_BreakoutNoFrameskip-v4_ckpt_steps_5000000.pth'
    should_record_video = True

    # Step 1: Prepare environment, replay buffer and schedule
    env = utils.get_env_wrapper(env_id, record_video=should_record_video)
    replay_buffer = ReplayBuffer(buffer_size)
    const_schedule = utils.ConstSchedule(epsilon_eval)  # lambda would also do - doing it like this for consistency

    # Step 2: Prepare the DQN model
    model_path = os.path.join(BINARIES_PATH, model_name)
    model_state = torch.load(model_path)
    utils.print_model_metadata(model_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN(env, number_of_actions=env.action_space.n, epsilon_schedule=const_schedule).to(device)
    dqn.load_state_dict(model_state["state_dict"], strict=True)
    dqn.eval()

    # Step 3: Evaluate the agent on a single episode
    print(f'{"*"*10} Starting the game. {"*"*10}')
    last_frame = env.reset()

    score = 0
    while True:
        replay_buffer.store_frame(last_frame)
        observation = replay_buffer.fetch_last_observation()  # fetch the observation, shape = (4, 84, 84) for Atari
        with torch.no_grad():
            action = dqn.epsilon_greedy(observation)  # act on the observation

        new_frame, reward, done, _ = env.step(action)  # send the action to the environment
        score += reward
        env.render()  # plot the current game frame

        if done:
            print(f'Episode over, score = {score}.')
            break

        last_frame = new_frame  # set the last frame to the newly acquired frame from the env
