import torch
import random
import numpy as np
from collections import deque # data structure that can add and remove items from both ends. to store the experience/memory
from snake_game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    
    def __init__(self) -> None:
        self.no_of_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate. 0 > gamma < 1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if len(memory) > MAX_MEMORY
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        direction_l = game.direction == Direction.LEFT  
        direction_r = game.direction == Direction.RIGHT
        direction_u = game.direction == Direction.UP
        direction_d = game.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (direction_r and game.is_collision(point_r)) or
            (direction_l and game.is_collision(point_l)) or
            (direction_u and game.is_collision(point_u)) or
            (direction_d and game.is_collision(point_d)),
            
            # Danger right
            (direction_u and game.is_collision(point_r)) or
            (direction_d and game.is_collision(point_l)) or
            (direction_l and game.is_collision(point_u)) or
            (direction_r and game.is_collision(point_d)),
            
            # Danger left
            (direction_d and game.is_collision(point_r)) or
            (direction_u and game.is_collision(point_l)) or
            (direction_r and game.is_collision(point_u)) or
            (direction_l and game.is_collision(point_d)),
            
            # Move direction
            direction_l,
            direction_r,
            direction_u,
            direction_d,
            
            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
            
        ]
        
        return np.array(state, dtype=int)
    
        
    
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # will pop the oldest memory if memory is full
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, game_overs = zip(*mini_sample) # TODO: read up on zip()
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
        
        # how i did it
        # for state, action, reward, next_state, game_over in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, game_over)
    
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)
    
    def get_action(self, state):
        # random moves: tradeoff exploration/exploitation
        self.epsilon = 80 - self.no_of_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) #will execute forward method in model.py
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
    

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)       
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)
        
        if game_over:
            # train long memory, plot result
            game.reset()
            agent.no_of_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.no_of_games, 'Score', score, 'Record', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.no_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            

if __name__ == '__main__':
    train()
