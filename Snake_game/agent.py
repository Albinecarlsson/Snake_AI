import torch
import random
import numpy as np

# deque is a duble ended queue
from collections import deque

from snake_game_AI import GameAI, dir, Cords, TILE_SIZE
from model import QTrainer, Linear_QNet
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005

class Agent:
    def __init__(self):
        self.nr_games = 0
        self.epsilon = 0.3 # randomness
        self.gamma = 0.7 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if the queue get full it will popleft()
        self.model = Linear_QNet(11,256,3) #a state have 11 params and we want an ansere of 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    
    def get_state(self, game):
        head = game.snake[0]
        cord_l = Cords(head.x - TILE_SIZE, head.y)
        cord_r = Cords(head.x + TILE_SIZE, head.y)
        cord_u = Cords(head.x, head.y - TILE_SIZE)
        cord_d = Cords(head.x, head.y + TILE_SIZE)
        
        dir_r = game.dir == dir.RIGHT
        dir_l = game.dir == dir.LEFT
        dir_u = game.dir == dir.UP
        dir_d = game.dir == dir.DOWN
        
        state = [
            # danger straight forward from snakes perspective
            (dir_r and game.is_collision(cord_r)) or
            (dir_l and game.is_collision(cord_l)) or
            (dir_u and game.is_collision(cord_u)) or
            (dir_d and game.is_collision(cord_d)),
            
            # danger right from snakes perspective
            (dir_u and game.is_collision(cord_r)) or
            (dir_d and game.is_collision(cord_l)) or
            (dir_l and game.is_collision(cord_u)) or
            (dir_r and game.is_collision(cord_d)),
            
            # danger left from snakes perspective
            (dir_d and game.is_collision(cord_r)) or
            (dir_u and game.is_collision(cord_l)) or
            (dir_r and game.is_collision(cord_u)) or
            (dir_l and game.is_collision(cord_d)),
            
            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #Food location
            game.apple.x < game.head.x, # apple to the left 
            game.apple.x > game.head.x, # apple to the right
            game.apple.y < game.head.y, # apple upwards
            game.apple.y > game.head.y  # apple downwards
        ]
        # return all true and false values as 1 respectivly 0 
        return np.array(state, dtype=int)
    
    def remeber(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # will pop left if memoery is full
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE) # return a random list of tuples of saved data
        else:
            sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*sample) # returns each values as a list of the same values (instead of using for loop)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self,state):
        # random moves: tradeoff between exploration and exploitation 
        self.epsilon = 80 - self.nr_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) #predict in tensorflow 
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_avg_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = GameAI()
    #traning loop
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        #get move
        final_move = agent.get_action(state_old)
        
        # preform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)
        
        # remember all of the params
        agent.remeber(state_old, final_move, reward, state_new, game_over)
        
        if game_over: 
            # train long memory (experieance replay, replay memory), plot resault
            game.reset()
            agent.nr_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                
            print('Game', agent.nr_games, 'score', score, 'Record', record)
            
            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.nr_games
            plot_avg_score.append(avg_score) 
            plot(plot_scores, plot_avg_score)      
        

if __name__ == '__main__':
    train()