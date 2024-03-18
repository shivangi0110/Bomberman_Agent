from collections import namedtuple, deque
from random import shuffle

import pickle
from typing import List
import os

import events as e
from .callbacks import state_to_features, ACTIONS
# from .rule_based_actions import act

import numpy as np
import torch
import torch.nn as nn

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
LAST_MAN_STANDING = "LAST_MAN_STANDING"
CLOSER_TO_ENEMY = "CLOSER_TO_ENEMY"
CLOSEST_TO_ENEMY = "CLOSEST_TO_ENEMY"
FURTHER_FROM_ENEMY = "FURTHER_FROM_ENEMY"
DANGER_ZONE_BOMB = "DANGER_ZONE_BOMB"
SAFE_CELL_BOMB = "SAFE_CELL_BOMB"
ALREADY_VISITED_EVENT = "ALREADY_VISITED_EVENT"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.training_states = []
    self.training_actions = []
    self.training_next_states = []
    self.training_rewards = []
    self.original_states = []

    self.gamma = 1.0
    self.learning_rate = 0.003

    self.imitation_loss = nn.CrossEntropyLoss().to(torch.device('cuda'))
    self.reinforcement_loss = nn.MSELoss().to(torch.device('cuda'))
    self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0

    if not os.path.isfile("iterations.txt"):
        self.iterations = 0.0
    else:
        self.iterations = np.array(np.loadtxt("iterations.txt"))[-1]

    self.loss_vec = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if old_game_state:
        if len(old_game_state["others"]) == 0:
            enemy_distance_old = 0
        else:
            _, _, _, pos_old = old_game_state["self"]
            pos_enemy_old = np.array([pos for _, _, _, pos in old_game_state['others']])
            enemy_distance_old = np.sum(np.abs(np.subtract(pos_enemy_old, pos_old)), axis=1).min()

    _, _, _, pos_current = new_game_state["self"]
    
    if len(new_game_state["others"]) == 0:
        enemy_distance_current = 0
    else:
        pos_enemy_current = np.array([pos for _, _, _, pos in new_game_state['others']])
        enemy_distance_current = np.sum(np.abs(np.subtract(pos_enemy_current, pos_current)), axis=1).min()
    
    # First round, set min distance to enemies
    if new_game_state["round"] == 1:
        self.min_enemy_distance = enemy_distance_current

    # Idea: Add your own events to hand out rewards
    if len(new_game_state["others"]) == 0:
        events.append(LAST_MAN_STANDING)
        self.logger.debug(f'Add game event {LAST_MAN_STANDING} in step {new_game_state["step"]}')

    if old_game_state and enemy_distance_current < enemy_distance_old:
        events.append(CLOSER_TO_ENEMY)
        self.logger.debug(f'Add game event {CLOSER_TO_ENEMY} in step {new_game_state["step"]}')

    if enemy_distance_current < self.min_enemy_distance:
        events.append(CLOSEST_TO_ENEMY)
        self.min_enemy_distance = enemy_distance_current
        self.logger.debug(f'Add game event {CLOSEST_TO_ENEMY} in step {new_game_state["step"]}')

    if old_game_state and enemy_distance_current > enemy_distance_old:
        events.append(FURTHER_FROM_ENEMY)
        self.logger.debug(f'Add game event {FURTHER_FROM_ENEMY} in step {new_game_state["step"]}')

    is_getting_bombed = False
    for (xb, yb), t in new_game_state["bombs"]:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if pos_current == (i, j):
                events.append(DANGER_ZONE_BOMB)
                self.logger.debug(f'Add game event {DANGER_ZONE_BOMB} in step {new_game_state["step"]}')
                is_getting_bombed = True
                break


            # if (0 < i < game_state['field'].shape[0]) and (0 < j < game_state['field'].shape[1]):
            #     bomb_map[i, j] = min(bomb_map[i, j], t)

    if not is_getting_bombed:
        events.append(SAFE_CELL_BOMB)
        self.logger.debug(f'Add game event {SAFE_CELL_BOMB} in step {new_game_state["step"]}')



    if self.iterations<500:

        self.logger.info("Imitation lerning interation.")
        # self.logger.info("training_states size")
        # self.logger.info(training_states.size())
        
        # f = self.model.forward(torch.tensor(state_to_features(old_game_state), dtype=torch.float))

        # target_action = rule_based_act(self, old_game_state)

        # target_index = ACTIONS.index("WAIT" if target_action==None else target_action)

        # target_vector = torch.tensor(np.eye(6)[target_index], dtype=torch.float)

        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

        rb_act = rule_based_act(self, old_game_state)

        if rb_act is not None:

            state_action_value = self.model.forward(state_to_features(old_game_state))

            target = torch.tensor(np.eye(6)[ACTIONS.index(rb_act)])
            # print(target)

            loss = self.imitation_loss(state_action_value, target)

            with open("loss_log.txt", "a") as loss_log:
                loss_log.write(str(loss.item()) + "\t")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

        # print(torch.mean(training_states[0]-training_states[-1]))
        # print(f)

        # target = target.view(-1, len(ACTIONS)).float()
        # f = f.view(-1, len(ACTIONS))

        # self.logger.info("f size")
        # self.logger.info(f.size())
        # self.logger.info("target size")
        # self.logger.info(target.size())

        # self.optimizer.zero_grad()
        # loss = self.imitation_loss(f, target_vector)
        # loss.backward()
        # # print(loss)
        # self.optimizer.step()

    else:

        self.logger.info("Reinforcement learning iteration.")

        # for i in range(len(training_actions)):
        # action_indices[i][ACTIONS.index(training_actions[i])] = True

        # q_states_max = torch.masked_select(self.model.forward(training_states), action_indices)
        f = torch.max(self.model.forward(torch.tensor(state_to_features(old_game_state), dtype=torch.float)))

        q_next_states_max = torch.max(self.model.forward(torch.tensor(state_to_features(new_game_state), dtype=torch.float)))
        # q_next_states_max = torch.max(torch.tensor(self.model.forward(state_to_features(new_game_state))), dim=1)
        q_target = reward_from_events(self, events) + self.gamma * q_next_states_max

        self.optimizer.zero_grad()
        loss = self.reinforcement_loss(f, q_target)
        loss.backward()
        self.optimizer.step()

        with open("loss_log.txt", "a") as loss_log:
            loss_log.write(str(np.mean(self.loss_vec)) + "\t")

    # with open("loss_log.txt", "a") as loss_log:
    #         loss_log.write(str(loss.item()) + "\t")

    # self.loss_vec.append(loss.item())


    # if self.visited_before[pos_current[0]][pos_current[1]] == 1:
    #     events.append(ALREADY_VISITED_EVENT)

    # self.visited_before = self.visited

    # self.visited = np.zeros((17, 17))
    # self.visited[pos_current[0]][pos_current[1]] = 1


    # state_to_features is defined in callbacks.py
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    # self.training_states.append(state_to_features(old_game_state))
    # self.training_actions.append(self_action)
    # self.training_next_states.append(state_to_features(new_game_state))
    # self.training_rewards.append(reward_from_events(self, events))
    # self.original_states.append(old_game_state)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # self.logger.info(f"Training_rewards: {self.training_rewards}")
    # self.logger.info(f"Training_states: {self.training_states}")

    # training_states = torch.tensor(np.array(self.training_states)).float()
    # training_next_states = torch.tensor(np.array(self.training_next_states)).float()
    # training_rewards = torch.tensor(np.array(self.training_rewards)).float()
    # training_actions = self.training_actions
    # original_states = self.original_states

    # print(torch.mean(training_states[0]-training_next_states[0]))
    # print()

    # action_indices = torch.zeros((training_states.shape[0], len(ACTIONS))).bool()\


    # if self.iterations<50000:

    #     # self.logger.info("Imitation lerning interation.")
    #     # self.logger.info("training_states size")
    #     # self.logger.info(training_states.size())
        
    #     f = self.model.forward(training_states)

    #     target_index = [rule_based_act(self, state) for state in original_states]

    #     rule_based_targets = [ACTIONS.index("WAIT" if t==None else t) for t in target_index]

    #     target = torch.tensor(np.array([np.eye(6)[target] for target in rule_based_targets]), dtype=torch.float)

    #     # print(torch.mean(training_states[0]-training_states[-1]))
    #     # print(f)

    #     # target = target.view(-1, len(ACTIONS)).float()
    #     # f = f.view(-1, len(ACTIONS))

    #     # self.logger.info("f size")
    #     # self.logger.info(f.size())
    #     # self.logger.info("target size")
    #     # self.logger.info(target.size())

    #     self.optimizer.zero_grad()
    #     loss = self.imitation_loss(f, target)
    #     loss.backward()
    #     # print(loss)
    #     self.optimizer.step()

    # else:

    #     self.logger.info("Reinforcement learning interation.")

    #     for i in range(len(training_actions)):
    #         action_indices[i][ACTIONS.index(training_actions[i])] = True

    #     q_states_max = torch.masked_select(self.model.forward(training_states), action_indices)
    #     q_next_states_max = torch.max(self.model.forward(training_next_states),
    #                                   dim=1)[0]
    #     q_target = training_rewards + self.gamma * q_next_states_max

    #     self.optimizer.zero_grad()
    #     loss = self.reinforcement_loss(q_target, q_states_max)
    #     loss.backward()
    #     self.optimizer.step()

    # Empty training data
    # self.training_states = []
    # self.training_actions = []
    # self.training_next_states = []
    # self.training_rewards = []
    # self.original_states = []

    self.iterations += 1

    self.logger.info(f"Interations completed: {self.iterations}")

    # self.visited = np.zeros((17, 17))
    # self.visited_before = np.zeros((17, 17))

    # self.exploration_rate = self.exploration_rate * self.EPS_DEC if self.exploration_rate > \
    #                                                                 self.EPS_MIN else self.EPS_MIN
    # Store the model
    with open("my_model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # with open("loss_log.txt", "a") as loss_log:
    #         loss_log.write(str(np.mean(self.loss_vec)) + "\t")

    with open("iterations.txt", "a") as iter:
            iter.write(str(self.iterations) + "\t")


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 1,
        e.KILLED_SELF: -0.5,
        e.CRATE_DESTROYED: 0.1,
        ALREADY_VISITED_EVENT: -0.05,
        LAST_MAN_STANDING: 1,
        CLOSER_TO_ENEMY: 0.002,
        CLOSEST_TO_ENEMY: 0.1,
        FURTHER_FROM_ENEMY: -0.002,
        DANGER_ZONE_BOMB: -0.000666,
        SAFE_CELL_BOMB: 0.002,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def rule_based_act(self, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    self.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1
    self.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
    self.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if self.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, self.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        self.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                self.bomb_history.append((x, y))

            return a

