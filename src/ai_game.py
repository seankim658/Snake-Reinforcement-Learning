# imports 
import pygame 
from pygame.locals import *
import random 
import time 
from enum import Enum, auto 
from collections import namedtuple
import numpy as np 

# start pygame 
pygame.init()
font = pygame.font.Font(None, 25)

# enumeration to denote current snake direction
class Direction(Enum):
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()

# virtualize game board as 2D grid of points 
Point = namedtuple('Point', 'x, y')
# size of each square in the game board
SQUARE_SIZE = 20

# game colors 
TEXT_COLOR = (0, 0, 0)
FOOD_COLOR = (171, 138, 143)
SNAKE_COLOR_MAIN = (204, 108, 67)
SNAKE_COLOR_ACCENT = (176, 93, 58)
BACKGROUND_COLOR = (255, 255, 255)

# game speed
SPEED = 20

class AISnake: 

    def __init__(self, width = 1280, height = 720):
        ''' Constructor.

        Parameters
        ----------
        width : int 
            Width of the game window. 
        height : int 
            Height of the game window.
        '''
        # dimensions of the game window (default to 1280x720)
        self.width = width 
        self.height = height 
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset_game() 

    def reset_game(self): 
        ''' Initiate a new game. 
        '''
        # initial direction for snake is going right 
        self.direction = Direction.RIGHT

        # create the head of the snake 
        self.head = Point(self.width / 2, self.height / 2)
        # initialize the snake to start with 3 body semgnets (including the head)
        self.snake = [self.head, Point(self.head.x - SQUARE_SIZE, self.head.y), Point(self.head.x - (2 * SQUARE_SIZE), self.head.y)]
        
        # hold the current game score 
        self.score = 0
        # hold the food piece 
        self.food = None
        # place initial food 
        self._generate_food()

        self.frame_iteration = 0
    
    def _generate_food(self):
        ''' Randomly place the food somewhere on the playing board. 
        '''
        x = random.randint(0, (self.width - SQUARE_SIZE) // SQUARE_SIZE) * SQUARE_SIZE
        y = random.randint(0, (self.height - SQUARE_SIZE) // SQUARE_SIZE) * SQUARE_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._generate_food()
    
    def step(self, input):
        ''' Define a game step by 1) collecting user (given by the model) input 2) responding to the input 
            3) checking if the game is over 4) continuing to the next step if the game is not over 
            and 5) updating the game.

            Parameters
            ----------
            input : List 
                The model's predicted best input. 
                [1, 0, 0] = keep going straight (no change in direction)
                [0, 1, 0] = turn right 
                [0, 0, 1] = turn left 

            Returns 
            -------
            game_result : boolean 
            score : int
            reward : int
        '''
        self.frame_iteration += 1
        # 1) collect user input (if any)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2) move according to user input 
        self._move(input)
        self.snake.insert(0, self.head)

        # 3) check if game is over (or if stuck in a long cycle where nothing has happened)
        reward = 0
        game_result = True
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_result = False 
            reward = -10
            return reward, game_result, self.score 

        # 4) start preparation for next step 
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._generate_food()
        else:
            self.snake.pop()

        # 5) update the game 
        self._update()
        self.clock.tick(SPEED)

        return reward, game_result, self.score
    
    def _is_collision(self, point = None):
        ''' Check if snake has hit itself or if it has hit a boundary wall.

        Parameters
        ----------
        point : Point 

        Returns
        -------
        boolean 
        '''
        if point is None:
            point = self.head
        if point.x > self.width - SQUARE_SIZE:
            return True 
        elif point.x < 0:
            return True 
        elif point.y > self.height - SQUARE_SIZE:
            return True 
        elif point.y < 0:
            return True 
        if point in self.snake[1:]:
            return True 
        return False 
    
    def _update(self):
        ''' Update the game graphics. 
        '''
        self.display.fill(BACKGROUND_COLOR)
        for segment in self.snake:
            pygame.draw.rect(self.display, SNAKE_COLOR_MAIN, pygame.Rect(segment.x, segment.y, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.rect(self.display, SNAKE_COLOR_ACCENT, pygame.Rect(segment.x + 4, segment.y + 4, 12, 12))
        pygame.draw.rect(self.display, FOOD_COLOR, pygame.Rect(self.food.x, self.food.y, SQUARE_SIZE, SQUARE_SIZE))
        text = font.render(f'Score: {self.score}', True, TEXT_COLOR)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def _move(self, input):
        ''' Move the snake.

        Parameters 
        ----------
        input : List 
            Predicted input from the model. 
            [1, 0, 0] = keep going straight (no change in direction)
            [0, 1, 0] = turn right 
            [0, 0, 1] = turn left 
        '''
        clockwise_directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        curr_direction = clockwise_directions.index(self.direction)

        if np.array_equal(input, [1, 0, 0]):
            new_direction = clockwise_directions[curr_direction]
        elif np.array_equal(input, [0, 1, 0]):
            new_direction = clockwise_directions[(curr_direction + 1) % 4]
        elif np.array_equal(input, [0, 0, 1]):
            new_direction = clockwise_directions[(curr_direction - 1) % 4]
        self.direction = new_direction

        x = self.head.x 
        y = self.head.y 
        if self.direction == Direction.RIGHT:
            x += SQUARE_SIZE
        elif self.direction == Direction.LEFT:
            x -= SQUARE_SIZE 
        elif self.direction == Direction.DOWN: 
            y += SQUARE_SIZE
        elif self.direction == Direction.UP:
            y -= SQUARE_SIZE
        self.head = Point(x, y)
    
if __name__ == '__main__':
    game = AISnake()

    while True:
        game_result, score = game.step()
        if game_result == False:
            break 
    print(f'Score: {score}')

    pygame.quit()