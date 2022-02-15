import numpy as np
import pygame
import sys
from random import randint
from pygame.locals import *

WIDTH = 700
HEIGHT = 500
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)


class Ball(pygame.sprite.Sprite):
    # This class represents a ball. It derives from the "Sprite" class in Pygame.
    def __init__(self, color, x, y):
        # Call the parent class (Sprite) constructor
        super().__init__()

        self.r = 6

        # Pass in the color of the ball, its width and height.
        # Set the background color and set it to be transparent
        self.image = pygame.Surface([self.r*2, self.r*2])
        self.image.fill(BLACK)				# RGB 24-bit color... 0 is Black
        self.image.set_colorkey(BLACK)

        pygame.draw.circle(self.image, color, [self.r, self.r], self.r)

        self.velocity = [randint(4, 8), randint(-8, 8)]
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
        self.velocity[1] = randint(-8, 8)


class Paddle(pygame.sprite.Sprite):
    def __init__(self, color, x, y):
        super().__init__()

        self.w = 10
        self.h = 100
        # self.score = 0
        self.goals = 0
        self.reward = 0
        self.hits = 0
        self.is_terminal_state = False
        self.for_is_terminal_state = 0
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
        # self.score = 0
        self.reward = 0
        self.hits = 0
        self.is_terminal_state = False
        self.goals = 0
        self.score = 0
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
    colors = {
        'BLACK': BLACK,
        'WHITE': WHITE,
        'GREY': GREY
    }

    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        pygame.init()
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("AI PONG")
        self.clock = pygame.time.Clock()

        xmargin = 20  # Margin from corner 20px

        # Position of the paddle and ball
        self.paddleA = Paddle(self.colors['WHITE'], xmargin, h//2)
        self.ball = Ball(self.colors['WHITE'], w//2, h//2)

        # list of all the sprites in the game.
        self.all_sprites = pygame.sprite.Group()
        self.all_sprites.add(self.paddleA)
        self.all_sprites.add(self.ball)

    def reset(self):
        xmargin = 20
        self.paddleA.reset(xmargin, self.h//2)
        self.ball.reset(self.w//2, self.h//2)

        # Return the initial positions
        a_observation = np.array(
            (self.paddleA.rect.centery, self.ball.rect.centery))

        return a_observation

    def render(self):
        # Display routine
        self.screen.fill(self.colors['BLACK'])
        pygame.draw.line(self.screen, self.colors['WHITE'], [
            self.w//2, 0], [self.w//2, self.w], 5)
        self.all_sprites.draw(self.screen)

        # Display scores:
        font = pygame.font.Font(None, 74)
        text = font.render(str(f'Goals: {self.paddleA.goals}'),
                           1, self.colors['WHITE'])
        self.screen.blit(text, (100, 10))
        text = font.render(
            str(f'Hits: {self.paddleA.hits}'), 1, self.colors['WHITE'])
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
        speed = 8
        self.paddleA.move((action-1) * speed)			# map from 0,1,2 to -1,0,1

        # Check if the ball is bouncing against any of the 4 walls:
        if self.ball.rect.x > self.w-self.ball.r*2:
            self.ball.velocity[0] = -self.ball.velocity[0]
            self.paddleA.score += 1
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

        # If limit reached then is_terminal_state...
        lim = 10
        if self.paddleA.score > lim:
            self.paddleA.is_terminal_state = True

        # return observations... 1 for each paddle and ball(only y-axis)
        a_observation = np.array(
            (
                self.paddleA.rect.centery,
                self.ball.rect.centery
            )
        )

        state = a_observation
        reward = self.paddleA.reward
        is_terminal_state = self.paddleA.is_terminal_state
        return (state, reward, is_terminal_state)

    def get_screen(self):
        # 2D (w,h) 32 bit color RGB 16777216
        arr = pygame.surfarray.array2d(self.screen)
        # ~ arr = pygame.surfarray.array3d(self.screen)	# 3D (w,h, 3) rgb
        return arr
        # ~ return np.dot(arr[...,:3], [0.299, 0.587, 0.114]) # RGB -> BW
