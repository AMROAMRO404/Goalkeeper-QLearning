import numpy as np

from environment import Pong
from helpers import DEFAULT_QTABLE, get_discrete_state, load_q_table, plot_args, q_learning_constants, qtable_path

# this should not be hardcoded but yet...
TABLE_SIZE = [30, 30]


def load_or_init_q_table(action_space, filename=DEFAULT_QTABLE):
    q_table = load_q_table(filename)
    if q_table is not None:
        print('using best ones..')
        return q_table
    return np.random.uniform(low=-2, high=0, size=(TABLE_SIZE + [action_space]))


def train(episodes=None, ball_speed=None, paddle_speed=None):
    env = Pong(ball_speed=ball_speed, paddle_speed=paddle_speed)
    LEARNING_RATE, DISCOUNT, DEFAULT_EPISODES, SHOW_EVERY, STATS_EVERY, PRINT_EVERY = q_learning_constants()
    EPISODES = episodes if episodes is not None else DEFAULT_EPISODES

    # Exploration settings
    epsilon = 0.1  # not a constant, going to be decayed
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

    action_space = env.action_space
    q_table = load_or_init_q_table(action_space)

    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    for episode in range(EPISODES):
        episode_reward = 0
        state = env.reset()
        is_terminal_state = False
        discrete_state = get_discrete_state(state, TABLE_SIZE)

        while not is_terminal_state:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, action_space)

            new_state, reward, is_terminal_state = env.step(action)
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state, TABLE_SIZE)

            if episode % SHOW_EVERY == 0:
                env.render()

            # If simulation did not end yet after last step - update Q table
            if not is_terminal_state:
                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]

                # Equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + \
                    LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                q_table[discrete_state + (action,)] = new_q

            # Simulation ended (for any reason) - if goal position is achieved - update Q value to highest
            elif env.paddleA.reward >= 10:
                q_table[discrete_state + (action,)] = 100

            discrete_state = new_discrete_state

        # Print progress every PRINT_EVERY episodes; save the Q-table less often (disk I/O).
        if not episode % PRINT_EVERY:
            print(f"episode {episode}/{EPISODES}  epsilon={epsilon:.4f}  "
                  f"episode_reward={episode_reward}  goals_this_episode={env.paddleA.goals}")
        if not episode % STATS_EVERY:
            np.save(qtable_path(episode), q_table)

        # Decaying is being done every episode if episode number is within decaying range
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        # Matplot visualize
        ep_rewards.append(episode_reward)
        if not episode % (STATS_EVERY // 10):
            average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))

    plot_args(aggr_ep_rewards)


if __name__ == "__main__":
    train()
