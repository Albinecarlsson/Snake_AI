import os
import math
import pygame
import random
from enum import Enum
from collections import namedtuple
import itertools
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'


class dir(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    
Cords = namedtuple('Cords', 'x, y')

pygame.init()


#Colours
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREY1 = (120, 120, 120)
GREY2 = (70, 70, 70)

# game variables
TILE_SIZE = 40
GAME_SPEED = 400


class GameAI:
    
    def __init__(self, WINDOW_WIDTH = 400, WINDOW_HEIGHT = 400):
        self.WINDOW_WIDTH = WINDOW_WIDTH
        self.WINDOW_HEIGHT = WINDOW_HEIGHT
        # initialize display
        self.display = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def px_to_idx(self,tile_unit):
        return int(tile_unit / TILE_SIZE)      
        
    def reset(self):
        # initialization of the game state.
        #snake stuff
        self.dir = dir.RIGHT
        self.head = Cords(self.WINDOW_WIDTH/2,self.WINDOW_HEIGHT/2)
        self.snake = [self.head,
                      Cords(self.head.x-TILE_SIZE,self.head.y),
                      Cords(self.head.x- (2*TILE_SIZE),self.head.y)]
        # apple stuff
        self.score = 0
        self.apple = None
        self.new_apple()
        self.frame_iteration = 0
     
    def to_px(self,tile_unit):
        return tile_unit * TILE_SIZE
        
    def new_apple(self):
            x = random.randint(0, (self.WINDOW_WIDTH-TILE_SIZE)//TILE_SIZE)*TILE_SIZE
            y = random.randint(0, (self.WINDOW_HEIGHT-TILE_SIZE)//TILE_SIZE)*TILE_SIZE
            self.apple = Cords(x,y)
            if self.apple in self.snake:
                self.new_apple()
            

    def is_collision(self, pt=None):
        # if we hit the boarder
        if pt is None:
            pt = self.head
            
        if (pt.x > self.WINDOW_WIDTH - TILE_SIZE or
            pt.y > self.WINDOW_HEIGHT - TILE_SIZE or
            pt.x < 0 or pt.y < 0):
            return True
        # if sanke hits snake
        if (pt in self.snake[1:]):
            return True
        return False
                
        
    def move(self, action):
        # [straight, right, left]
        clock_wise = [dir.RIGHT, dir.DOWN, dir.LEFT, dir.UP]
        idx = clock_wise.index(self.dir)
        
        if np.array_equal(action, [1,0,0]): # go straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0,1,0]): # turn right 
            new_dir = clock_wise[(idx+1) % 4]
        else:
            new_dir = clock_wise[(idx-1) % 4] # turn left
        self.dir = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.dir == dir.UP:
            y-= TILE_SIZE  
        elif self.dir == dir.DOWN:
            y+= TILE_SIZE  
        elif self.dir == dir.LEFT:
            x-= TILE_SIZE  
        elif self.dir == dir.RIGHT:
            x+= TILE_SIZE
        self.head = Cords(x,y)
        
        
    def get_distance(self,apple,head):
        x_power = np.power(self.px_to_idx(apple.x) - self.px_to_idx(head.x), 2)
        y_power = np.power(self.px_to_idx(apple.y) - self.px_to_idx(head.y), 2)
        return (np.sqrt(x_power + y_power))

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        #move the snake
        self.move(action)
        self.snake.insert(0,self.head)
    
        #check if Game Over
        reward = 0
        GAME_OVER = False
        if self.is_collision() or self.frame_iteration > 50*len(self.snake):
            GAME_OVER = True
            reward = -10
            return reward, GAME_OVER, self.score

        # place new food
        if self.head == self.apple:
            self.score+=1
            reward = 10
            self.new_apple()
        else:
            self.snake.pop()
            
        self.update_ui()
        self.clock.tick(GAME_SPEED)
        return reward, GAME_OVER, self.score
            
            
            
    def update_ui(self):
        # refill background colours
        bg_colors = itertools.cycle([GREY1, GREY2])
        for y in range(0, self.WINDOW_HEIGHT, TILE_SIZE):
            for x in range(0, self.WINDOW_WIDTH, TILE_SIZE):
                rect = (x, y, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.display, next(bg_colors), rect)
            next(bg_colors)
        
        #paint the snake
        for bit in self.snake:
            pygame.draw.rect(self.display, BLUE1, 
                             pygame.Rect(bit.x, bit.y, TILE_SIZE, TILE_SIZE))
            if (bit == self.head):
                pygame.draw.rect(self.display, BLUE2, 
                             pygame.Rect(bit.x+4, bit.y+4, 12, 12))
       
        # paint apple 
        pygame.draw.rect(self.display, RED, [
                        self.apple.x, self.apple.y, TILE_SIZE, TILE_SIZE]) 
       
        # update score 
        font = pygame.font.SysFont('Monaco', 42)
        score_img = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(score_img, [0, 0])
        pygame.display.flip()