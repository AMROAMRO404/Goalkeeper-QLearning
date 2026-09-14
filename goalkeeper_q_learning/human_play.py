import sys

import pygame
from pygame.locals import K_s, K_w, QUIT

from environment import Pong

WIDTH = 700
HEIGHT = 500


def get_human_action():
    # Same action encoding as the trained agent: 0=up, 1=none, 2=down
    keys = pygame.key.get_pressed()
    if keys[K_w]:
        return 0
    if keys[K_s]:
        return 2
    return 1


def play_human(ball_speed=None, paddle_speed=None):
    env = Pong(w=WIDTH, h=HEIGHT, ball_speed=ball_speed, paddle_speed=paddle_speed)
    env.reset()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        env.step(get_human_action())
        env.render()


if __name__ == '__main__':
    play_human()
