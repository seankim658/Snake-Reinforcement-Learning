# imports 
import torch 
import random 
import numpy as np 
from collections import deque
from ai_game import AISnake, Direction, Point 
from model import QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000 
LEARNING_RATE = 0.001
EPSILON_THRESHOLD = 80

class GameAgent:

    def __init__(self): 
        ''' Constructor. 
        '''
        self.n_games = 0
        self.epsilon = 0 # how much we want to explore (controls randomness)
        self.gamma = 0.9 # discount factor (helps to balance long term and short term rewards)
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)
    
    def get_state(self, game):
        ''' Capture the game state. 

        Returns
        -------
        np.array 
            The game state in the following format.
                [danger straight, danger right, danger left,
                direction left, direction right, direction up, direction down,
                food left, food right, food up, food down]   
        ''' 
        head = game.snake[0]
        left_point = Point(head.x - game.square_size, head.y)
        right_point = Point(head.x + game.square_size, head.y)
        up_point = Point(head.x, head.y - game.square_size)
        down_point = Point(head.x, head.y + game.square_size)

        game_direction = game.direction
        direction_left = (game_direction == Direction.LEFT)
        direction_right = (game_direction == Direction.RIGHT)
        direction_up = (game_direction == Direction.UP)
        direction_down = (game_direction == Direction.DOWN)

        state = [
            # danger straight 
            (direction_right and game.is_collision(right_point)) or 
            (direction_left and game.is_collision(left_point)) or 
            (direction_up and game.is_collision(up_point)) or 
            (direction_down and game.is_collision(down_point)),

            # danger right 
            (direction_right and game.is_collision(down_point)) or
            (direction_left and game.is_collision(up_point)) or 
            (direction_up and game.is_collision(right_point)) or 
            (direction_down and game.is_collision(left_point)), 

            # danger left 
            (direction_right and game.is_collision(up_point)) or
            (direction_left and game.is_collision(down_point)) or
            (direction_up and game.is_collision(left_point)) or
            (direction_down and game.is_collision(right_point)), 

            # move direction
            direction_left,
            direction_right,
            direction_up,
            direction_down, 

            # food direction 
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right 
            game.food.y < game.head.y, # food up 
            game.food.y > game.head.y # food down 
        ]        

        return np.array(state, dtype = int)

    def remember(self, state, input, reward, next_state, game_result):
        ''' Add the game state to memory. Since the deque was initialized with a max length, 
            if the max length has been reached, an element is removed from the left side (popleft).
        '''
        self.memory.append((state, input, reward, next_state, game_result)) 

    def train_long_memory(self):
        ''' Train the model's long term memory after one game cycle is complete.
        '''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory
        states, inputs, rewards, future_states, game_results = zip(*mini_sample)
        self.trainer.train_step(states, inputs, rewards, future_states, game_results)

    def train_short_memory(self, state, input, reward, next_state, game_result):
        ''' Train the model's short term memory for one game step. 
        '''
        self.trainer.train_step(state, input, reward, next_state, game_result)

    def get_action(self, state):
        '''
        '''
        self.epsilon = EPSILON_THRESHOLD - self.n_games # the more games played, the less exploration should be done
        next_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            next_move[move] = 1
        else:
            curr_state = torch.tensor(state, dtype = torch.float)
            pred = self.model(curr_state)
            move = torch.argmax(pred).item()
            next_move[move] = 1
        
        return next_move 

def train():
    '''
    '''
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = GameAgent
    game = AISnake
    while True:
        # get current state 
        curr_state = agent.get_state(game)

        # get move based on current state 
        new_move = agent.get_action(curr_state)

        # act on the new move and get new game state 
        reward, game_result, score = game.step(new_move) 
        new_state = agent.get_state(game)

        # train short memory 
        agent.train_short_memory(curr_state, new_move, reward, new_state, game_result)

        # remember 
        agent.remember(curr_state, new_move, reward, new_state, game_result)

        if game_result == False:
            # reset the game 
            game.reset_game()

            # train long memory and plot the results 
            agent.n_games += 1
            agent.train_long_memory()

            # check is the last game set a new high schore 
            if score > record:
                record = score 
                agent.model.saveI()
            
            # print outcomes 
            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')

            # TODO: plot 

if __name__ == '__main__':
    train()