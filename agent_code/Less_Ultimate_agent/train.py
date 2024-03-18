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
    self.imit_cutoff = 100000000000
    # self.imit_cutoff = -1

    if not os.path.isfile("epochs.txt"):
        self.epochs = 0.0
    else:
        self.epochs = np.array(np.loadtxt("epochs.txt"))[-1]

    if self.epochs > self.imit_cutoff:
        print('Carrying out Reinforcement Learning')
    else:
        print('Carrying out Imitation Learning')

    print(f'Training epoch: {self.epochs}')

    self.batch_rewards = []

    self.model.gamma = 0.999
    self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    self.current_explosion_map_t = np.zeros((17, 17), dtype=np.double)
    self.old_explosion_map_t = np.zeros((17, 17), dtype=np.double)

def update_target_network(self):
    self.target_network.load_state_dict(self.model.state_dict())

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    a = act_rule(old_game_state)
    old_features = state_to_features(old_game_state, self.current_explosion_map_t, self.old_explosion_map_t)
    new_features = state_to_features(new_game_state, old_game_state['explosion_map'], self.current_explosion_map_t)

    if self.epochs < self.imit_cutoff:
        self.model.imitation_train_step(old_features, self_action, a)
    
    check_events(self, old_game_state, self_action, new_game_state, events)

    self.old_explosion_map_t = self.current_explosion_map_t
    self.current_explosion_map_t = old_game_state['explosion_map']

    self.replay_buffer.push(old_features, ACTIONS.index(self_action), reward_from_events(events), new_features)
    

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    if len(self.replay_buffer.buffer) >= self.batch_size and self.epochs >= self.imit_cutoff:
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.model.device)
        action_batch = torch.LongTensor(batch[1]).to(self.model.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.model.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.model.device)

        batch_total_reward = np.mean(batch[2])
        with open("batch_rewards.txt", "a") as rewards_file:
            rewards_file.write(str(batch_total_reward) + "\t")

        q_values = self.model.forward(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_network.forward(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.model.gamma * next_q_values

        loss = self.model.reinf_loss(q_values, expected_q_values.unsqueeze(1)).to(self.model.device)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        self.steps += 1

        if self.steps % 100 == 0:
            update_target_network(self)

        with open("loss_log.txt", "a") as loss_log:
            loss_log.write(str(loss.item()) + "\t")
        with open("loss_log_reinf.txt", "a") as loss_log:
            loss_log.write(str(loss.item()) + "\t")

    if self.epochs % 1000 == 0:
        torch.save(self.model.state_dict(), 'my-saved-model_reinf_'+str(int(self.epochs))+'.pt')

    self.epochs += 1
    torch.save(self.model.state_dict(), 'my-saved-model.pt')

    with open("epochs.txt", "a") as ep:
        ep.write(str(self.epochs) + "\t")

def reward_from_events(events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.COIN_FOUND: 2,
        e.KILLED_OPPONENT: 10,
        e.KILLED_SELF: -10,
        # e.MOVED_DOWN: 0.1,
        # e.MOVED_UP: 0.1,
        # e.MOVED_RIGHT: 0.1,
        # e.MOVED_LEFT: 0.1,
        e.BOMB_DROPPED: -2,
        e.INVALID_ACTION: -0.5,
        # e.WAITED: -0.5,
        e.CRATE_DESTROYED: 2,
        e.SURVIVED_ROUND: 20,
        e.GOT_KILLED: -10,
        CLOSER_TO_ENEMY: -1,
        FURTHER_FROM_ENEMY: 1,
        DANGER_ZONE: -1,
        SAFE_ZONE: 0.5,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    return reward_sum

def check_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    pos_enemy_old = np.array([pos[3] for pos in old_game_state['others']])
    if old_game_state and len(pos_enemy_old) > 0:
        pos_old = old_game_state["self"][3]
        enemy_distance_old = np.sum(np.abs(np.subtract(pos_enemy_old, pos_old)), axis=1).min()

    pos_current = new_game_state["self"][3]
    pos_enemy_current = np.array([pos[3] for pos in new_game_state['others']])
    
    if len(pos_enemy_current) > 0:
        enemy_distance_current = np.sum(np.abs(np.subtract(pos_enemy_current, pos_current)), axis=1).min()

        if old_game_state and enemy_distance_current < enemy_distance_old:
            events.append(CLOSER_TO_ENEMY)
            self.logger.debug(f'Add game event {CLOSER_TO_ENEMY} in step {new_game_state["step"]}')

        if old_game_state and enemy_distance_current > enemy_distance_old:
            events.append(FURTHER_FROM_ENEMY)
            self.logger.debug(f'Add game event {FURTHER_FROM_ENEMY} in step {new_game_state["step"]}')
    
    is_getting_bombed = False
    for (xb, yb), t in new_game_state['bombs']:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if pos_current == (i, j):
                is_getting_bombed = True

    if is_getting_bombed:
        events.append(DANGER_ZONE)
        self.logger.debug(f'Add game event {DANGER_ZONE} in step {new_game_state["step"]}')
    else:
        events.append(SAFE_ZONE)
        self.logger.debug(f'Add game event {SAFE_ZONE} in step {new_game_state["step"]}')