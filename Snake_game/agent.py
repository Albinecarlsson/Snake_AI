import torch
import random
import numpy as np
from collections import deque
from snake_game_AI import GameAI, dir, Cords, TILE_SIZE
from model import Trainer, NeuralNetwork
from helper import plot

# set torch default to GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#
#   Change this load path to where the same file is saved.
#
Load_path = 'C:/Users/albin/OneDrive/Dokument/Github/Snake_AI/model/extra_vision_predict_move.pth'

# Memory for the deque
MAX_MEMORY = 100_000

BATCH_SIZE = 1000
LR = 0.005

class Agent:
    def __init__(self):
        self.nr_games = 0
        self.epsilon = 1000 # randomness variable
        self.gamma = 0.7 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if the queue get full it will popleft()
        self.model = NeuralNetwork(20,256,3) # state have 20 params and we want an answer of 3
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma)
        
    # Get state
    def get_state(self, game):
        head = game.snake[0]
        # Save Squares of length 1 from the head 
        cord_l = Cords(head.x - TILE_SIZE, head.y)
        cord_r = Cords(head.x + TILE_SIZE, head.y)
        cord_u = Cords(head.x, head.y - TILE_SIZE)
        cord_d = Cords(head.x, head.y + TILE_SIZE)
        
        # Save Squares of length 2 from the head   
        cord_l2 = Cords(head.x - 2 * TILE_SIZE, head.y)
        cord_r2 = Cords(head.x + 2 * TILE_SIZE, head.y)
        cord_u2 = Cords(head.x, head.y - 2 * TILE_SIZE)
        cord_d2 = Cords(head.x, head.y + 2 * TILE_SIZE)
        
        # Save Squares of length 3 from the head 
        cord_l3 = Cords(head.x - 3 * TILE_SIZE, head.y)
        cord_r3 = Cords(head.x + 3 * TILE_SIZE, head.y)
        cord_u3 = Cords(head.x, head.y - 3 * TILE_SIZE)
        cord_d3 = Cords(head.x, head.y + 3 * TILE_SIZE)
        
        # Save Squares of length 4 from the head 
        cord_l4 = Cords(head.x - 4 * TILE_SIZE, head.y)
        cord_r4 = Cords(head.x + 4 * TILE_SIZE, head.y)
        cord_u4 = Cords(head.x, head.y - 4 * TILE_SIZE)
        cord_d4 = Cords(head.x, head.y + 4 * TILE_SIZE)
        
        # Save What direction the snake is going in
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
            
             # danger 2 square straight forward from snakes perspective
            (dir_r and game.is_collision(cord_r2)) or
            (dir_l and game.is_collision(cord_l2)) or
            (dir_u and game.is_collision(cord_u2)) or
            (dir_d and game.is_collision(cord_d2)),
            
             # danger 3 square straight forward from snakes perspective
            (dir_r and game.is_collision(cord_r3)) or
            (dir_l and game.is_collision(cord_l3)) or
            (dir_u and game.is_collision(cord_u3)) or
            (dir_d and game.is_collision(cord_d3)),
            
             # danger 4 square straight forward from snakes perspective
            (dir_r and game.is_collision(cord_r4)) or
            (dir_l and game.is_collision(cord_l4)) or
            (dir_u and game.is_collision(cord_u4)) or
            (dir_d and game.is_collision(cord_d4)),
            
            # danger 1 square right from snakes perspective
            (dir_u and game.is_collision(cord_r)) or
            (dir_d and game.is_collision(cord_l)) or
            (dir_l and game.is_collision(cord_u)) or
            (dir_r and game.is_collision(cord_d)),
            # danger 2 square right from snakes perspective
            (dir_u and game.is_collision(cord_r2)) or
            (dir_d and game.is_collision(cord_l2)) or
            (dir_l and game.is_collision(cord_u2)) or
            (dir_r and game.is_collision(cord_d2)),
            # danger 3 square right from snakes perspective
            (dir_u and game.is_collision(cord_r3)) or
            (dir_d and game.is_collision(cord_l3)) or
            (dir_l and game.is_collision(cord_u3)) or
            (dir_r and game.is_collision(cord_d3)),
            # danger 4 square right from snakes perspective
            (dir_u and game.is_collision(cord_r4)) or
            (dir_d and game.is_collision(cord_l4)) or
            (dir_l and game.is_collision(cord_u4)) or
            (dir_r and game.is_collision(cord_d4)),
            
            # danger 1 square left from snakes perspective
            (dir_d and game.is_collision(cord_r)) or
            (dir_u and game.is_collision(cord_l)) or
            (dir_r and game.is_collision(cord_u)) or
            (dir_l and game.is_collision(cord_d)),
            # danger 2 square left from snakes perspective
            (dir_d and game.is_collision(cord_r2)) or
            (dir_u and game.is_collision(cord_l2)) or
            (dir_r and game.is_collision(cord_u2)) or
            (dir_l and game.is_collision(cord_d2)),
            # danger 3 square left from snakes perspective
            (dir_d and game.is_collision(cord_r3)) or
            (dir_u and game.is_collision(cord_l3)) or
            (dir_r and game.is_collision(cord_u3)) or
            (dir_l and game.is_collision(cord_d3)),
            # danger 4 square left from snakes perspective
            (dir_d and game.is_collision(cord_r4)) or
            (dir_u and game.is_collision(cord_l4)) or
            (dir_r and game.is_collision(cord_u4)) or
            (dir_l and game.is_collision(cord_d4)),
            
            # Current direction the snake Moves in
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # The direction of the apple from snakes perspective 
            game.apple.x < game.head.x,
            game.apple.x > game.head.x,
            game.apple.y < game.head.y,
            game.apple.y > game.head.y
        ]
        # converts the boolean array to an int array before returning 
        return np.array(state, dtype=int)
    # Saves into memory 
    def remeber(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # will pop left if memoery is full
    
    # Trains the "long term" memory by selecting a random batch from memory if we have that many enteries
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
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
                # Calculate distance and move to the left 
                dist_left = (game.get_distance(game.apple, Cords(game.head.x - TILE_SIZE,game.head.y)))
                col_left = game.is_collision(Cords(game.head.x - TILE_SIZE,game.head.y)) 
                # Calculate distance and move to the right                 
                dist_right = (game.get_distance(game.apple, Cords(game.head.x + TILE_SIZE,game.head.y)))
                col_right = game.is_collision(Cords(game.head.x + TILE_SIZE,game.head.y)) 
                # Calculate distance and move to the upwards 
                dist_up = (game.get_distance(game.apple, Cords(game.head.x,game.head.y - TILE_SIZE)))
                col_up = game.is_collision(Cords(game.head.x,game.head.y - TILE_SIZE)) 
                # Calculate distance and move to the downwards 
                dist_down = (game.get_distance(game.apple, Cords(game.head.x,game.head.y + TILE_SIZE)))
                col_down = game.is_collision(Cords(game.head.x,game.head.y + TILE_SIZE)) 

                # Select move based of what direction you are moving in where we check collisions aswell
                if game.dir == dir.RIGHT:
                    move = np.argmin([dist_right,dist_down,dist_up])
                    if col_right and move == 0: move += 1
                    if col_down and move == 1: move += 1
                    if col_up and move == 2: move = 0
                elif game.dir == dir.LEFT:
                    move = np.argmin([dist_left,dist_up,dist_down])
                    if col_left and move == 0: move += 1
                    if col_up and move == 1: move += 1
                    if col_down and move == 2: move = 0
                elif game.dir == dir.UP:
                    move = np.argmin([dist_up,dist_right,dist_left])
                    if col_up and move == 0: move += 1
                    if col_right and move == 1: move += 1
                    if col_left and move == 2: move = 0
                elif game.dir == dir.DOWN:
                    move = np.argmin([dist_down,dist_left,dist_right])
                    if col_down and move == 0: move += 1
                    if col_left and move == 1: move += 1
                    if col_right and move == 2: move = 0
                else: # probably redundant
                    move = random.randint(0, 2)
                final_move[move] = 1
                print(final_move)
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
                agent.model.save(file_name='extra_vision_predict_move.pth')
            
            # if yo wanna see staticstic for each  run uncomment line below
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
    # Loading from local path where i have saved the model file
    agent.model.load(Load_path)

    #traning loop
    while True:
        # get current state
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
            #print('Game', agent.nr_games, 'score', score, 'Record', record)
            
            # Calculate values for ploting average score and score for current game
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