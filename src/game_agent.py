# imports 
import torch 
import random 
import numpy as np 
from collections import deque
from ai_game import AISnake, Direction, Point 

MAX_MEMORY = 100_00
BATCH_SIZE = 1000 
LEARNING_RATE = 0.001

class GameAgent:

    def __init__(self):
        ''' Constructor. 
        '''
        self.n_games = 0
        self.epsilon = 0 # how much we want to explor (controls randomness)
        self.gamma = 0.9 # discount factor 
        self.memory = deque(maxlen = MAX_MEMORY)
    
    def get_state(self, game):
        pass 

    def remember(self, state, input, reward, next_state, done):
        pass 

    def train_long_memory(self):
        pass 

    def train_short_memory(self, state, action, reward, next_state, done):
        pass 

    def get_action(self, state):
        pass 

def train():
    pass 

if __name__ == '__main__':
    train()