import numpy as np
import pygame
import sys
from random import randint
from pygame.locals import *

from helpers import DEFAULT_BALL_SPEED, DEFAULT_PADDLE_SPEED

WIDTH = 600
HEIGHT = 440


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class Ball(pygame.sprite.Sprite):
    # This class represents a ball. It derives from the "Sprite" class in Pygame.
    def __init__(self, color, x, y, speed=DEFAULT_BALL_SPEED):
        # Call the parent class (Sprite) constructor
        super().__init__()

        self.r = 6

        # Pass in the color of the ball, its width and height.
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([self.r*2, self.r*2])
        self.image.fill(BLACK)				# RGB 24-bit color... 0 is Black
        self.image.set_colorkey(BLACK)

        pygame.draw.circle(self.image, color, [self.r, self.r], self.r)

        self.x_speed_range = (speed - 2, speed + 2)
        self.y_speed_range = (-(speed + 2), speed + 2)
        self.velocity = [randint(*self.x_speed_range), randint(*self.y_speed_range)]
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    def reset(self, x, h):
        self.rect.center = (x, randint(self.r, h-self.r))
        self.bounce()

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]

    def bounce(self):
        self.velocity[0] = -self.velocity[0]
        self.velocity[1] = randint(*self.y_speed_range)


class Paddle(pygame.sprite.Sprite):
    def __init__(self, color, x, y):
        super().__init__()

        self.w = 10
        self.h = 100
        self.goals = 0
        self.reward = 0
        self.previous_reward = 0
        self.hits = 0
        self.is_terminal_state = False
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([self.w, self.h])
        self.image.fill(0)					# 0 is Black
        self.image.set_colorkey(0)			# make Black color transparent

        pygame.draw.rect(self.image, color, [0, 0, self.w, self.h])
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    def collided(self, other):
        self.is_terminal_state = True
        self.reward += 10
        self.hits += 1
        other.bounce()

    def reset(self, x, y):
        self.reward = 0
        self.previous_reward = 0
        self.hits = 0
        self.is_terminal_state = False
        self.goals = 0
        self.rect.center = (x, y)

    def move(self, y):
        self.rect.y += y
        if self.rect.y < 0:
            self.rect.y = 0
        if self.rect.y > HEIGHT-self.h:
            self.rect.y = HEIGHT-self.h


class Pong():
    FPS = 30  # Frame per second
    action_space = 3			# 3 actions... up,down,none

    def __init__(self, w=WIDTH, h=HEIGHT, ball_speed=None, paddle_speed=None):
        self.w = w
        self.h = h
        self.paddle_speed = DEFAULT_PADDLE_SPEED if paddle_speed is None else paddle_speed
        pygame.init()
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("AI PONG")
        self.clock = pygame.time.Clock()

        self.xmargin = 20  # Margin from corner 20px

        # Position of the paddle and ball
        self.paddleA = Paddle(WHITE, self.xmargin, h//2)
        self.ball = Ball(WHITE, w//2, h//2,
                          speed=DEFAULT_BALL_SPEED if ball_speed is None else ball_speed)

        # list of all the sprites in the game.
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.paddleA)
        self.all_sprites.add(self.ball)

    def reset(self):
        self.paddleA.reset(self.xmargin, self.h//2)
        self.ball.reset(self.w//2, self.h)

        # Return the initial positions
        a_observation = np.array(
            (
                self.paddleA.rect.centery,
                self.ball.rect.centery
            )
        )

        return a_observation

    def render(self):
        # Display routine
        self.screen.fill(BLACK)
        pygame.draw.line(self.screen, WHITE, [
            self.w//2, 0], [self.w//2, self.w], 5)
        self.all_sprites.draw(self.screen)

        # Display scores:
        font = pygame.font.Font(None, 74)
        text = font.render(str(f'Goals: {self.paddleA.goals}'),   1, WHITE)
        self.screen.blit(text, (100, 10))
        text = font.render(
            str(f'Hits: {self.paddleA.hits}'), 1, WHITE)
        self.screen.blit(text, (410, 10))
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def step(self, action):
        self.all_sprites.update()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Moving the paddles according to action given... -1, 0 or 1
        self.paddleA.move((action-1) * self.paddle_speed)	# map from 0,1,2 to -1,0,1

        # Check if the ball is bouncing against any of the 4 walls:
        if self.ball.rect.x > self.w-self.ball.r*2:
            self.ball.velocity[0] = -self.ball.velocity[0]
        if self.ball.rect.x < 0:
            self.paddleA.goals += 1
            self.paddleA.reward -= 1
            self.ball.velocity[0] = -self.ball.velocity[0]
        if self.ball.rect.y > self.h-self.ball.r*2:
            self.ball.velocity[1] = -self.ball.velocity[1]
        if self.ball.rect.y < 0:
            self.ball.velocity[1] = -self.ball.velocity[1]

        # Detect collisions between the ball and the paddles
        if pygame.sprite.collide_rect(self.ball, self.paddleA) and self.ball.velocity[0] < 0:
            self.paddleA.collided(self.ball)

        # return observations... 1 for each paddle and ball(only y-axis)
        a_observation = np.array(
            (
                self.paddleA.rect.centery,
                self.ball.rect.centery
            )
        )

        state = a_observation
        # Reward for this transition only, not the episode's running total
        reward = self.paddleA.reward - self.paddleA.previous_reward
        self.paddleA.previous_reward = self.paddleA.reward
        is_terminal_state = self.paddleA.is_terminal_state
        return (state, reward, is_terminal_state)
