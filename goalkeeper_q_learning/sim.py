import numpy as np

from environment import Pong
from helpers import DEFAULT_QTABLE, get_discrete_state, load_q_table
from train import TABLE_SIZE


def sim(filename=DEFAULT_QTABLE, episodes=None, ball_speed=None, paddle_speed=None):
    """Watch a trained agent play against the simulated ball (no webcam needed)."""
    q_table = load_q_table(filename)
    if q_table is None:
        print('q-table not found:', filename)
        return

    env = Pong(ball_speed=ball_speed, paddle_speed=paddle_speed)
    episode = 0
    while episodes is None or episode < episodes:
        state = env.reset()
        is_terminal_state = False
        while not is_terminal_state:
            discrete_state = get_discrete_state(state, TABLE_SIZE)
            action = np.argmax(q_table[discrete_state])
            state, _, is_terminal_state = env.step(action)
            env.render()
        episode += 1


if __name__ == "__main__":
    sim()
