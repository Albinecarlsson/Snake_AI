import torch
import random
import numpy as np
from collections import deque

from snake_game_AI import GameAI, dir, Cords, TILE_SIZE
from model import Trainer, NeuralNetwork
from helper import plot

# set torch default to GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

Load_path = 'C:/Users/albin/OneDrive/Dokument/Github/Snake_AI/model/base_case.pth'
# Memory for the deque
MAX_MEMORY = 100_000

BATCH_SIZE = 1000
LR = 0.005

class Agent:
    def __init__(self):
        self.nr_games = 0
        self.epsilon = 1000 # randomness
        self.gamma = 0.7 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if the queue get full it will popleft()
        self.model = NeuralNetwork(11,512,3) #a state have 11 params and we want an ansere of 3
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma)
        
    
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
            # danger 1 square straight forward from snakes perspective
            (dir_r and game.is_collision(cord_r)) or
            (dir_l and game.is_collision(cord_l)) or
            (dir_u and game.is_collision(cord_u)) or
            (dir_d and game.is_collision(cord_d)),
            
            # danger 1 square right from snakes perspective
            (dir_u and game.is_collision(cord_r)) or
            (dir_d and game.is_collision(cord_l)) or
            (dir_l and game.is_collision(cord_u)) or
            (dir_r and game.is_collision(cord_d)),
            
            # danger 1 square left from snakes perspective
            (dir_d and game.is_collision(cord_r)) or
            (dir_u and game.is_collision(cord_l)) or
            (dir_r and game.is_collision(cord_u)) or
            (dir_l and game.is_collision(cord_d)),
             
            # Current direction the snake Moves in
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # The direction of the apple from snakes perspective 
            game.apple.x < game.head.x, # apple to the left 
            game.apple.x > game.head.x, # apple to the right
            game.apple.y < game.head.y, # apple upwards
            game.apple.y > game.head.y  # apple downwards
        ]
        # converts the boolean array to an int array before returning
        return np.array(state, dtype=int)
    
    # Saves into memory 
    def remeber(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # will pop left if memoery is full
    
    # Trains the "long term" memory by selecting a random batch from memory if we have that many enteries
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE) # return a random list of tuples of saved data
        else:
            sample = self.memory
         # Returns each values as a list of the same values (instead of using for loop)    
        states, actions, rewards, next_states, dones = zip(*sample) 
        self.trainer.train_step(states, actions, rewards, next_states, dones)
     
    # Trains the memory on current move       
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    
    # Get move to make
    def get_move(self,state,game,training=True):
        final_move = [0,0,0] # [straight, right, left]
        if training:
            # tradeoff between exploration and exploitation where we do the move that brings us closer to the apple
            # if not predict state with Neural network
            if random.randint(0,5*self.nr_games) < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                # make move with the neural network
                state = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
        else:
            # make move with the neural network                
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state)
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
        # get current state
        current_state = agent.get_state(game)
        #get move
        final_move = agent.get_move(current_state,game)
        
        # preform move and get new state
        reward, game_over, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        #train short memory
        agent.train_short_memory(current_state, final_move, reward, new_state, game_over)
        
        # remember all of the params
        agent.remeber(current_state, final_move, reward, new_state, game_over)
        
        if game_over: 
            # Restart game and update number iterations
            # train long memory (experieance replay, replay memory), plot resault
            game.reset()
            agent.nr_games += 1
            # Update epsilon to decrease the exploration
            if agent.epsilon > 1:
                agent.epsilon = agent.epsilon * 0.999
            else:
                agent.epsilon = 1
            # train long term memory    
            agent.train_long_memory()
            
            # Might wanna change so it save each 200 iterations 
            # If we reach new high score save
            if score > record:
                record = score
                agent.model.save(file_name='base_case.pth')
            
            # if yo wanna see staticstic for each  run un comment line below
            #print('Game', agent.nr_games, 'score', score, 'Record', record)
            
            # Calculate values for ploting average score and score for current game
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
    agent.model.load(Load_path)

 #traning loop
    while True:
        # get old state
        current_state = agent.get_state(game)
        
        #get move
        final_move = agent.get_move(current_state,game,training)
        
        # preform move and get new state
        reward, game_over, score = game.play_step(final_move)
        new_state = agent.get_state(game)
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
    training = False
    if training:
        train()
    else:
        evaluate()