import pickle
import os
import random
import torch
from collections import namedtuple, deque
from typing import List
import events as e
from .state_to_features import state_to_features
import numpy as np
from .act_rule import act_rule

# Events
CLOSER_TO_ENEMY = "CLOSER_TO_ENEMY"
FURTHER_FROM_ENEMY = "FURTHER_FROM_ENEMY"
DANGER_ZONE = "DANGER_ZONE"
SAFE_ZONE = "SAFE_ZONE"

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup_training(self):
    if not os.path.isfile("epochs.txt"):
        self.epochs = 0.0
    else:
        self.epochs = np.array(np.loadtxt("epochs.txt"))[-1]

    print(f'Training epoch: {self.epochs}')

    self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    self.current_explosion_map_t = np.zeros((17, 17), dtype=np.double)
    self.old_explosion_map_t = np.zeros((17, 17), dtype=np.double)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    a = act_rule(old_game_state)
    old_features = state_to_features(old_game_state, self.current_explosion_map_t, self.old_explosion_map_t)
    self.model.imitation_train_step(old_features, self_action, a)
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    
    if self.epochs % 1000 == 0:
        torch.save(self.model.state_dict(), 'my-saved-model_'+str(int(self.epochs))+'.pt')

    self.epochs += 1
    torch.save(self.model.state_dict(), 'my-saved-model.pt')

    with open("epochs.txt", "a") as ep:
        ep.write(str(self.epochs) + "\t")
