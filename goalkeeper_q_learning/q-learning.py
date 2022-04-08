from turtle import shape
import numpy as np
import pygame
import sys
import cv2
from cv2 import imshow
from pygame.locals import *
from enviroment import Pong, Paddle, Ball
from helpers import plot_args, q_learning_constants

# create the environment...
env = Pong()

LEARNING_RATE, DISCOUNT, EPISODES, SHOW_EVERY, STATS_EVERY, PLAY = q_learning_constants()

PLAY = True

# Exploration settings
epsilon = 0.1 	# not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# There are 2 observation space, so table will be 2D
TABLE_SIZE = [30, 30]       			# this should not hard coded but yet...
action_space = env.action_space			# 3 actions... -1,0,1


def get_discrete_state(state):
    # Normalize the state... position value / table_size
    discrete_state = state / TABLE_SIZE
    # we use this tuple to look up the 3 Q values for the available actions in the q-table
    return tuple(discrete_state.astype(np.int))


# Train agent ....
if not PLAY:
    # use the best q-table, if available...
    try:
        q_table = np.load(
            "goalkeeper_q_learning/qtables/qtable_0_e4000.npy")
        # print(np.shape(q_table))
        print('using best ones..')
    except:
        q_table = np.random.uniform(
            low=-2, high=0, size=(TABLE_SIZE + [action_space]))  # [30, 30, 3]
    # For stats
    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
    for episode in range(EPISODES):
        episode_reward = 0
        s = env.reset()
        is_terminal_state = False
        discrete_state = get_discrete_state(s)
        while not is_terminal_state:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(q_table[discrete_state])
            else:
                # Get random action
                action = np.random.randint(0, action_space)

            new_state, reward, is_terminal_state = env.step(action)
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state)

            if (episode % SHOW_EVERY == 0):
                env.render()
            # If simulation did not end yet after last step - update Q table
            if not is_terminal_state:
                # Maximum possible Q value in next step (for new state)
                max_future_q = np.max(q_table[new_discrete_state])

                # Current Q value (for current state and performed action)
                current_q = q_table[discrete_state + (action,)]

                # Equation for a new Q value for current state and action
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                # Update Q table with new Q value
                q_table[discrete_state + (action,)] = new_q

            # Simulation ended (for any reson) - if goal position is achived - update Q value to highest
            else:
                if (env.paddleA.reward >= 10):
                    q_table[discrete_state + (action,)] = 100

            discrete_state = new_discrete_state
        # Save the Q-table
        if not episode % STATS_EVERY:
            print("hiii")
            np.save(
                "goalkeeper_q_learning/qtables/qtable_0_e{}.npy".format(episode), q_table)

        # Decaying is being done every episode if episode number is within decaying range
        if (END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING):
            epsilon -= epsilon_decay_value

        # Matplot visualize
        ep_rewards.append(episode_reward)
        if not episode % (STATS_EVERY//10):
            average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
            #~ print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
    plot_args(aggr_ep_rewards)


# After Training, time to play...
try:
    q_table_a = np.load(
        "goalkeeper_q_learning/qtables/qtable_0_e4000.npy")

    print('file found')
    found = True
except:
    print('file not found')
    found = False

s = env.reset()
maxix = 0 
minix=400
maxiy = 0
miniy=400
print(s)
#while found:
if found:
    x_medium = 0
    y_medium = 0
    border_x = 5
    border_y = 20
    border_color = (255, 0, 0)
    ball_border_color = (0, 255, 0)
    thikness = 2
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        rows, cols, _ = frame.shape

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        ORANGE_MIN = np.array([10, 100, 120], np.uint8)
        ORANGE_MAX = np.array([25, 255, 255], np.uint8)

        mask = cv2.inRange(hsv_frame, ORANGE_MIN, ORANGE_MAX)
        countours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        countours = sorted(countours,key=lambda x: cv2.contourArea(x),reverse=True)

        rows = rows - border_y
        cols = cols - border_x

        cv2.line(frame,  (border_x, border_y), (border_x, rows),border_color, thikness)

        cv2.line(frame,  (border_x, border_y), (cols, border_y),border_color, thikness)

        cv2.line(frame,  (cols, border_y), (cols, rows),border_color, thikness)

        cv2.line(frame,  (border_x, rows), (cols, rows),border_color, thikness)

        for cnt in countours:
            (x, y, w, h) = cv2.boundingRect(cnt)

            x_medium = (x + w//2)
            y_medium = (y + h//2)

            if (x_medium >= border_x and x_medium <= cols) and (y_medium >= border_y and y_medium <= rows):
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                            ball_border_color, thikness)
                cv2.line(frame,  (x_medium, border_y),
                        (x_medium, rows), ball_border_color, thikness)
                cv2.line(frame, (border_x, y_medium),
                        (cols, y_medium), ball_border_color, thikness)

            # the values that goes to RL model
            x_medium = x_medium - border_x
            y_medium = y_medium - border_y

            if (x_medium < border_x):
                x_medium = 0

            if (y_medium < border_y):
                y_medium = 0

            if (x_medium > cols - border_x):
                x_medium = cols - border_x

            if (y_medium > rows - border_y):
                y_medium = rows - border_y

            print("x_medium = ", x_medium, ", y_medium = ", y_medium)

            break

        cv2.imshow("Frame", frame)
        ################
        maxix = max(s[0],maxix)#390
        maxiy = max(s[1],maxiy)#440
        minix = min(s[0],minix)#50
        miniy = min(s[1],miniy)#-2
        print(s)#first para for keeper location , second for ball x location 
        s[1]=y_medium
        state_a = get_discrete_state(s)
        #print(state_a) state a is related to the discretized value of what i dont know 
        action_a = np.argmax(q_table_a[state_a])
        #print(action_a) 2 ==> Down , 1 ==> do nothing , 0 ==>UP
        s, _, _ = env.step(action_a)
        env.render()
        #print(minix , miniy)
