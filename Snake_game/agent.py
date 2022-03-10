from pickle import TRUE
import torch
import random
import numpy as np

# deque is a duble ended queue
from collections import deque

from snake_game_AI import GameAI, dir, Cords, TILE_SIZE
from model import QTrainer, Linear_QNet
from helper import plot

MAX_MEMORY = 1_000_000
BATCH_SIZE = 1000
LR = 0.001

# set torch default to GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Agent:
    def __init__(self):
        self.nr_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.7 # discount rate 
        self.memory = deque(maxlen=MAX_MEMORY) # if the queue get full it will popleft()
        self.model = Linear_QNet(100,256,3)#.to(device) #a state have 14 params and we want an answere of 3 with a hidden layer of 256
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    
    def get_state(self, game):
        head = game.snake[0]
        cord_l = Cords(head.x - TILE_SIZE, head.y)
        cord_r = Cords(head.x + TILE_SIZE, head.y)
        cord_u = Cords(head.x, head.y - TILE_SIZE)
        cord_d = Cords(head.x, head.y + TILE_SIZE)
        
        cord_ll = Cords(head.x - 2 * TILE_SIZE, head.y)
        cord_rr = Cords(head.x + 2 * TILE_SIZE, head.y)
        cord_uu = Cords(head.x, head.y - 2 * TILE_SIZE)
        cord_dd = Cords(head.x, head.y + 2 * TILE_SIZE)
        
        dir_r = game.dir == dir.RIGHT
        dir_l = game.dir == dir.LEFT
        dir_u = game.dir == dir.UP
        dir_d = game.dir == dir.DOWN
        
        state = [       
            #Move direction
            #dir_l,
            #dir_r,
            #dir_u,
            #dir_d, 
            #Apple location
            #game.apple.x < game.head.x, # apple to the left 
            #game.apple.x > game.head.x, # apple to the right
            #game.apple.y < game.head.y, # apple upwards
            #game.apple.y > game.head.y  # apple downwards

            # gameboard   
        ]
        [state.append(pos) for row in game.board for pos in row]
        
        # return all true and false values as 1 respectivly 0   array size 15        
        return np.array(state, dtype=int)
    
    def remeber(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # will pop left if memoery is full since its a deque
    
    
    
    def train_long_memory(self):
        # select batchsize if we have batchsize in memory
        if len(self.memory) > BATCH_SIZE:
            # return a random list of tuples of saved data
            sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            # othervise take all of the memory
            sample = self.memory
        # returns each values as a list of the same values (instead of using for loop)
        states, actions, rewards, next_states, dones = zip(*sample) 
        
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_move(self,state,game):
        # random moves: tradeoff between exploration and exploitation  where we train with random moves
        self.epsilon = 80 - self.nr_games # decreasing with the nr of traning runs
        final_move = [0,0,0]
        # randomness in the moves
        if random.randint(0, 200) < self.epsilon:
            dist = game.get_distance(game.apple, game.head)
            dist_left = game.get_distance(game.apple, Cords(game.head.x - TILE_SIZE,game.head.y))
            dist_right = game.get_distance(game.apple, Cords(game.head.x + TILE_SIZE,game.head.y))
            dist_up = game.get_distance(game.apple, Cords(game.head.x,game.head.y - TILE_SIZE))
            dist_down = game.get_distance(game.apple, Cords(game.head.x,game.head.y + TILE_SIZE))
            if game.dir == dir.RIGHT:
                move = np.argmin(dist_up,dist_right,dist_down)
            elif game.dir == dir.LEFT:
                move = np.argmin(dist_down,dist_left,dist_up)
            elif game.dir == dir.UP:
                move = np.argmin(dist_left,dist_up,dist_right)
            elif game.dir == dir.DOWN:
                move = np.argmin(dist_right,dist_down,dist_left)
            else:
                move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # if not predict state with Neural network
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
    game = GameAI(traning=True)
    #traning loop
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        #get move
        final_move = agent.get_move(state_old,game)
        
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
            
            # if yo wanna see staticstic for each  run un comment line below
            print('Game', agent.nr_games, 'score', score, 'Record', record)
            
            # create variables for ploting avg score and score for gamenr

            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.nr_games
            plot_avg_score.append(avg_score) 
            plot(plot_scores, plot_avg_score)      
        

if __name__ == '__main__':
    train()