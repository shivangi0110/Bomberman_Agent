import os
import random
import pickle
import torch
from collections import deque
from .model import DQN
import numpy as np
from .act_rule import act_rule
from .state_to_features import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def setup(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0
    self.current_round = 0
    self.global_counter = 0
    self.model = DQN()

    if self.train:
        self.model.device = torch.device('cuda')
        if not os.path.isfile("my-saved-model.pt"):
            self.logger.info("Setting up model from scratch.")
            print("Setting up model from scratch.")
        else:
            self.logger.info("Loading model from saved state.")
            print("Loading model from saved state.")
            self.model.load_state_dict(torch.load('my-saved-model.pt'))
        self.model.to(self.model.device)
    else:
        self.model.device = torch.device('cpu')
        if not os.path.isfile("my-saved-model.pt"):
            self.logger.info("Setting up model from scratch.")
            print("Setting up model from scratch.")
        else:
            self.logger.info("Loading model from saved state.")
            print("Loading model from saved state.")
            self.model.load_state_dict(torch.load('my-saved-model.pt', map_location=torch.device('cpu')))
        self.model.to(self.model.device)

    if next(self.model.parameters()).is_cuda:
        print('Running on cuda.')
    else:
        print('Running on cpu.')

    self.target_network = DQN()
    self.target_network.load_state_dict(self.model.state_dict())
    self.target_network.eval()
    self.replay_buffer = ReplayBuffer(10000)
    self.batch_size = 64
    self.steps = 0

    self.current_explosion_map_a = np.zeros((17, 17), dtype=np.double)
    self.old_explosion_map_a = np.zeros((17, 17), dtype=np.double)

def act(self, game_state: dict) -> str:
    random_prob = 0.0

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[0, 0, 0, 0, 1, 0])

    self.logger.debug("Querying model for action.")
    f = state_to_features(game_state, self.current_explosion_map_a, self.old_explosion_map_a)
    input = torch.FloatTensor(f[np.newaxis, :]).to(self.model.device)

    self.old_explosion_map_a = self.current_explosion_map_a
    self.current_explosion_map_a = game_state['explosion_map']
    a = self.model.forward(input)[0]

    res = ACTIONS[a.argmax()]
    return res
