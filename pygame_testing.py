# imports 
import pygame 
from pygame.locals import *

import random 
import time

from enum import Enum 
from collections import namedtuple

# start pygame 
pygame.init() 
font = pygame.font.Font(None, 25)

# simple enumeration to denote current snake direction 
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# virtualize game board as a 2D plane 
Point = namedtuple('Point', 'x, y')

# game colors 
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BACKGROUND_COLOR = (0, 0, 0)

# game speed 
SPEED = 20
# size of each square in the game board 
SQUARE_SIZE = 20

class Snake:

    def __init__(self, width = 640, height = 480):
        ''' Constructor. 

        Parameters
        ----------
        width : int
            Width of the game window. 
        height : int
            Height of the game window.
        '''
        # dimensions of the game window (default to 640x480)
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # initial direction for snake is going up 
        self.direction = Direction.RIGHT 
        # create the head of the snake 
        self.head = Point(self.width / 2, self.height / 2)
        # initialize the snake to start with 3 body segments (including the head)
        self.snake = [self.head, Point(self.head.x - SQUARE_SIZE, self.head.y), Point(self.head.x  - (2 * SQUARE_SIZE), self.head.y)]
        # hold the current game score 
        self.score = 0
        self.food = None 
        self._generate_food()
    
    def _generate_food(self):
        ''' Randomly place the food somewhere on the playing board. 
        '''
        x = random.randint(0, (self.width - SQUARE_SIZE) // SQUARE_SIZE) * SQUARE_SIZE
        y = random.randint(0, (self.height - SQUARE_SIZE) // SQUARE_SIZE) * SQUARE_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._generate_food()

    def step(self):
        ''' Define a game step by 1) collecting user input 2) responding to the input 
            3) checking if the game is over 4) continuing to the next step if the game is not over  
            and 5) updating the game 
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # 1) collect user input 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        # 2) move according to user input 
        self._move(self.direction)
        self.snake.insert(0, self.head)
        # 3) check if game is over 
        game_result = True 
        if self._is_collision():
            game_result = False 
            return game_result, self.score 
        # 4) start preparation for next step 
        if self.head == self.food:
            self.score += 1
            self._generate_food()
        else:
            self.snake.pop()
        # 5) update the game 
        self._update()
        self.clock.tick(SPEED)

        return game_result, self.score 
    
    def _is_collision(self):
        ''' Check if snake has hit itself or if it has hit a boundary wall. 
        '''
        if self.head.x > self.width - SQUARE_SIZE:
            return True   
        elif self.head.x < 0:
            return True  
        elif self.head.y > self.height - SQUARE_SIZE:
            return True  
        elif self.head.y < 0:
            return True  
        if self.head in self.snake[1:]:
            return True  
        return False 
    
    def _update(self):
        ''' Update the game UI. 
        '''
        self.display.fill(BACKGROUND_COLOR)
        for segment in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(segment.x, segment.y, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(segment.x + 4, segment.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, SQUARE_SIZE, SQUARE_SIZE))
        text = font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def _move(self, direction):
        ''' Move the snake.

        Parameters
        ----------
        direction : enum 
            Current direction the snake is moving. 
        '''
        x = self.head.x
        y = self.head.y 
        if direction == Direction.RIGHT:
            x += SQUARE_SIZE
        elif direction == Direction.LEFT:
            x -= SQUARE_SIZE
        elif direction == Direction.DOWN:
            y += SQUARE_SIZE
        elif direction == Direction.UP:
            y -= SQUARE_SIZE
        self.head = Point(x, y)

if __name__ == '__main__':
    game = Snake()

    while True:
        game_result, score = game.step()
        if game_result == False:
            break
    print(f'Score: {score}')
    
    pygame.quit() 