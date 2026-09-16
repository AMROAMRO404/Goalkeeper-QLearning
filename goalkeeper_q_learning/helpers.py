from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

QTABLES_DIR = Path(__file__).resolve().parent / "qtables"
DEFAULT_QTABLE = "qtable_0_e4000.npy"
DEFAULT_BALL_SPEED = 10
DEFAULT_PADDLE_SPEED = 12
DEFAULT_BAUD = 9600


def plot_args(aggr_ep_rewards):
    plt.plot(aggr_ep_rewards['ep'],
             aggr_ep_rewards['avg'], label="average rewards")
    plt.plot(aggr_ep_rewards['ep'],
             aggr_ep_rewards['max'], label="max rewards")
    plt.plot(aggr_ep_rewards['ep'],
             aggr_ep_rewards['min'], label="min rewards")
    plt.savefig(QTABLES_DIR / "paddle_0.png")
    plt.clf()


def q_learning_constants():
    LEARNING_RATE = 0.1
    DISCOUNT = 1
    EPISODES = 5001
    SHOW_EVERY = 1
    STATS_EVERY = 100
    PRINT_EVERY = 1
    return LEARNING_RATE, DISCOUNT, EPISODES, SHOW_EVERY, STATS_EVERY, PRINT_EVERY


def get_discrete_state(state, table_size):
    # Normalize the state... position value / table_size
    discrete_state = state / table_size
    # we use this tuple to look up the Q values for the available actions in the q-table
    return tuple(discrete_state.astype(int))


def qtable_path(episode):
    return QTABLES_DIR / f"qtable_0_e{episode}.npy"


def load_q_table(filename=DEFAULT_QTABLE):
    try:
        return np.load(QTABLES_DIR / filename)
    except FileNotFoundError:
        return None
