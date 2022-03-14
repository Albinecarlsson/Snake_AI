import torch
import random
import numpy as np

# deque is a duble ended queue
from collections import deque

from snake_game_AI import GameAI, dir, Cords, TILE_SIZE
from model import QTrainer, Linear_QNet
from helper import plot

# set torch default to GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.005

class Agent:
    def __init__(self):
        self.nr_games = 0
        self.epsilon = 1000 # randomness
        self.gamma = 0.7 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if the queue get full it will popleft()
        self.model = Linear_QNet(11,512,3) #a state have 11 params and we want an ansere of 3
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
    
    def get_state(self, game):
        head = game.snake[0]
        cord_l = Cords(head.x - TILE_SIZE, head.y)
        cord_r = Cords(head.x + TILE_SIZE, head.y)
        cord_u = Cords(head.x, head.y - TILE_SIZE)
        cord_d = Cords(head.x, head.y + TILE_SIZE)
        cord_l2 = Cords(head.x - 2 * TILE_SIZE, head.y)
        cord_r2 = Cords(head.x + 2 * TILE_SIZE, head.y)
        cord_u2 = Cords(head.x, head.y - 2 * TILE_SIZE)
        cord_d2 = Cords(head.x, head.y + 2 * TILE_SIZE)
        
        cord_l3 = Cords(head.x - 3 * TILE_SIZE, head.y)
        cord_r3 = Cords(head.x + 3 * TILE_SIZE, head.y)
        cord_u3 = Cords(head.x, head.y - 3 * TILE_SIZE)
        cord_d3 = Cords(head.x, head.y + 3 * TILE_SIZE)
        
        cord_l4 = Cords(head.x - 4 * TILE_SIZE, head.y)
        cord_r4 = Cords(head.x + 4 * TILE_SIZE, head.y)
        cord_u4 = Cords(head.x, head.y - 4 * TILE_SIZE)
        cord_d4 = Cords(head.x, head.y + 4 * TILE_SIZE)
        
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
            
            # danger straight forward from snakes perspective
        #    (dir_r and game.is_collision(cord_r2)) or
        #    (dir_l and game.is_collision(cord_l2)) or
        #    (dir_u and game.is_collision(cord_u2)) or
        #    (dir_d and game.is_collision(cord_d2)),
            
            # danger straight forward from snakes perspective
        #    (dir_r and game.is_collision(cord_r3)) or
        #    (dir_l and game.is_collision(cord_l3)) or
        #    (dir_u and game.is_collision(cord_u3)) or
        #    (dir_d and game.is_collision(cord_d3)),
            
            # danger straight forward from snakes perspective
        #    (dir_r and game.is_collision(cord_r4)) or
        #    (dir_l and game.is_collision(cord_l4)) or
        #    (dir_u and game.is_collision(cord_u4)) or
        #    (dir_d and game.is_collision(cord_d4)),
            
            # danger right from snakes perspective
            (dir_u and game.is_collision(cord_r)) or
            (dir_d and game.is_collision(cord_l)) or
            (dir_l and game.is_collision(cord_u)) or
            (dir_r and game.is_collision(cord_d)),
            # danger right from snakes perspective
        #    (dir_u and game.is_collision(cord_r2)) or
        #    (dir_d and game.is_collision(cord_l2)) or
        #    (dir_l and game.is_collision(cord_u2)) or
        #    (dir_r and game.is_collision(cord_d2)),
            # danger right from snakes perspective
        #    (dir_u and game.is_collision(cord_r3)) or
        #    (dir_d and game.is_collision(cord_l3)) or
        #    (dir_l and game.is_collision(cord_u3)) or
        #    (dir_r and game.is_collision(cord_d3)),
            # danger right from snakes perspective
        #    (dir_u and game.is_collision(cord_r4)) or
        #    (dir_d and game.is_collision(cord_l4)) or
        #    (dir_l and game.is_collision(cord_u4)) or
        #    (dir_r and game.is_collision(cord_d4)),
            
            # danger left from snakes perspective
            (dir_d and game.is_collision(cord_r)) or
            (dir_u and game.is_collision(cord_l)) or
            (dir_r and game.is_collision(cord_u)) or
            (dir_l and game.is_collision(cord_d)),
            # danger left from snakes perspective
        #    (dir_d and game.is_collision(cord_r2)) or
        #    (dir_u and game.is_collision(cord_l2)) or
        #    (dir_r and game.is_collision(cord_u2)) or
        #    (dir_l and game.is_collision(cord_d2)),
            # danger left from snakes perspective
        #    (dir_d and game.is_collision(cord_r3)) or
        #    (dir_u and game.is_collision(cord_l3)) or
        #    (dir_r and game.is_collision(cord_u3)) or
        #    (dir_l and game.is_collision(cord_d3)),
            # danger left from snakes perspective
        #    (dir_d and game.is_collision(cord_r4)) or
        #    (dir_u and game.is_collision(cord_l4)) or
        #    (dir_r and game.is_collision(cord_u4)) or
        #    (dir_l and game.is_collision(cord_d4)),
            
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
    
    def get_move(self,state,game,training=True):
        final_move = [0,0,0] # [straight, right, left]
        if training:
            # random moves: tradeoff between exploration and exploitation  where we train with random moves
            # if not predict state with Neural network
            if random.randint(0,5*self.nr_games) < self.epsilon:
            #    dist_left = (game.get_distance(game.apple, Cords(game.head.x - TILE_SIZE,game.head.y)))
            #    col_left = game.is_collision(Cords(game.head.x - TILE_SIZE,game.head.y)) 
                
            #    dist_right = (game.get_distance(game.apple, Cords(game.head.x + TILE_SIZE,game.head.y)))
            #    col_right = game.is_collision(Cords(game.head.x + TILE_SIZE,game.head.y)) 
                
            #    dist_up = (game.get_distance(game.apple, Cords(game.head.x,game.head.y - TILE_SIZE)))
            #    col_up = game.is_collision(Cords(game.head.x,game.head.y - TILE_SIZE)) 
                
            #    dist_down = (game.get_distance(game.apple, Cords(game.head.x,game.head.y + TILE_SIZE)))
            #    col_down = game.is_collision(Cords(game.head.x,game.head.y + TILE_SIZE)) 

            #    if game.dir == dir.RIGHT:
            #        move = np.argmin([dist_right,dist_down,dist_up])
            #        if col_right and move == 0: move += 1
            #        if col_down and move == 1: move += 1
            #        if col_up and move == 2: move = 0
            #    elif game.dir == dir.LEFT:
            #        move = np.argmin([dist_left,dist_up,dist_down])
            #        if col_left and move == 0: move += 1
            #        if col_up and move == 1: move += 1
            #        if col_down and move == 2: move = 0
            #    elif game.dir == dir.UP:
            #        move = np.argmin([dist_up,dist_right,dist_left])
            #        if col_up and move == 0: move += 1
            #        if col_right and move == 1: move += 1
            #        if col_left and move == 2: move = 0
            #    elif game.dir == dir.DOWN:
            #        move = np.argmin([dist_down,dist_left,dist_right])
            #        if col_down and move == 0: move += 1
            #        if col_left and move == 1: move += 1
            #        if col_right and move == 2: move = 0
            #    else: # probably redundant
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                # make move with the neural network
                state = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
                print(final_move)
        else:
            # make move with the neural network                
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            print(final_move)
            
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
            if agent.epsilon > 1:
                agent.epsilon = agent.epsilon * 0.999
            else:
                agent.epsilon = 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save(file_name='base_case.pth')
            
            # if yo wanna see staticstic for each  run un comment line below
            #print('Game', agent.nr_games, 'score', score, 'Record', record)
            
            # create variables for ploting avg score and score for gamenr
            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.nr_games
            plot_avg_score.append(avg_score) 
            plot(plot_scores, plot_avg_score)          
        
def evaluate(training=False):
    plot_scores = []
    plot_avg_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = GameAI(training=False)
    agent.model.load()

 #traning loop
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        #get move
        final_move = agent.get_move(state_old,game,training)
        
        # preform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        if game_over: 
            # plot resault
            game.reset()
            agent.nr_games += 1        
            
            # if yo wanna see staticstic for each  run un comment line below
            print('Game', agent.nr_games, 'score', score, 'Record', record)
            
            
            # create variables for ploting avg score and score for gamenr
            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.nr_games
            plot_avg_score.append(avg_score) 
            plot(plot_scores, plot_avg_score)      


if __name__ == '__main__':
    training = True
    if training:
        train()
    else:
        evaluate()