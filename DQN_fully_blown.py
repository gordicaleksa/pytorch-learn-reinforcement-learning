import argparse
from itertools import count
# todo: what is force used for in Monitor?


def train_dqn(training_config):

    for step_cnt in count():
        if env.monitor.get_total_steps() >= training_config['num_of_training_steps']:
            break

        collect_experience()

        if step_cnt > training_config['start_learning']:
            dqn_learn_from_experience()


def get_training_args():
    parser = argparse.ArgumentParser()

    # Training related
    parser.add_argument("--num_of_training_steps", type=int, help="number of training env steps", default=200000000)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    return training_config


if __name__ == '__main__':
    # Train the graph attention network (GAT)
    train_dqn(get_training_args())

