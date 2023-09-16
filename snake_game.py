import pygame
import random
from enum import Enum # allows for enumeration of a set of names for values
from collections import namedtuple # assigns a meaning to each position in a tuple and allows for more readable code
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Reset
# Reward
# Play(action) -> direction
# GameIteration
# isCollision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20
SPEED = 10

# RGB colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0) # for game over
BLUE1 = (0, 0, 255) # for snake
BLUE2 = (0, 100, 255) # for snake
YELLOW = (255, 255, 0) # for food

class SnakeGameAI:
    def __init__(self, w=640, h=480) -> None:
        self.w = w
        self.h = h
        
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()
        
        
        
    def reset(self):
        
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x-BLOCK_SIZE, self.head.y), Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    
    def play_step(self, action):
        self.frame_iteration += 1
        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.K_ESCAPE:
                pygame.quit()
                quit()
                
        # Move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # Check if game over.boundary or eat itself
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop() # remove the tail. because we inserted the head, we have to remove the a block from the tail
            
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
            
        # hits boundary
        if pt.x > self.w-BLOCK_SIZE or pt.x < 0 or pt.y > self.h-BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, BLOCK_SIZE*0.6, BLOCK_SIZE*0.6))
            
        pygame.draw.rect(self.display, YELLOW, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render('Score: ' + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip() # updates the entire surface. without this, nothing will be displayed

    def _move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index_of_current_direction = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[index_of_current_direction] # GO straight
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (index_of_current_direction+1)%4
            new_direction = clock_wise[next_index] # turn right, R -> D -> L -> U
        else: # [0, 0, 1]
            next_index = (index_of_current_direction-1)%4
            new_direction = clock_wise[next_index] # turn left, R -> U -> L -> D
            
        self.direction = new_direction
            
        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
            
        self.head = Point(x, y)