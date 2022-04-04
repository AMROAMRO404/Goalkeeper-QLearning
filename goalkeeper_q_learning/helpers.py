from matplotlib import pyplot as plt


def plot_args(aggr_ep_rewards):
    plt.plot(aggr_ep_rewards['ep'],
             aggr_ep_rewards['avg'], label="average rewards")
    plt.plot(aggr_ep_rewards['ep'],
             aggr_ep_rewards['max'], label="max rewards")
    plt.plot(aggr_ep_rewards['ep'],
             aggr_ep_rewards['min'], label="min rewards")
    #~ plt.show()
    plt.savefig("goalkeeper_q_learning/qtables/paddle_0.png")
    plt.clf()


def q_learning_constants():
    LEARNING_RATE = 0.00001
    DISCOUNT = 1
    EPISODES = 4001
    SHOW_EVERY = 500
    STATS_EVERY = 100
    PLAY = True
    return LEARNING_RATE, DISCOUNT, EPISODES, SHOW_EVERY, STATS_EVERY, PLAY
