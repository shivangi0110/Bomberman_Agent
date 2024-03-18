from collections import deque
from random import shuffle
import numpy as np

def look_for_targets(free_space, start, targets, logger=None):
    if not targets:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while frontier:
        current = frontier.pop(0)
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            best = current
            break
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger:
        logger.debug(f'Suitable target found at {best}')
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]

def act_rule(game_state):
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

    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (d not in others) and
                (d not in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles:
        valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles:
        valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles:
        valid_actions.append('UP')
    if (x, y + 1) in valid_tiles:
        valid_actions.append('DOWN')
    if (x, y) in valid_tiles:
        valid_actions.append('WAIT')
    if bombs_left > 0:
        valid_actions.append('BOMB')

    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    targets = [t for t in targets if t not in bomb_xys]

    free_space = arena == 0
    d = look_for_targets(free_space, (x, y), targets)
    if d == (x, y - 1):
        action_ideas.append('UP')
    if d == (x, y + 1):
        action_ideas.append('DOWN')
    if d == (x - 1, y):
        action_ideas.append('LEFT')
    if d == (x + 1, y):
        action_ideas.append('RIGHT')

    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    if len(others) > 0 and min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others) <= 1:
        action_ideas.append('BOMB')
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            if (yb > y):
                action_ideas.append('UP')
            if (yb < y):
                action_ideas.append('DOWN')
            action_ideas.extend(['LEFT', 'RIGHT'])
        if (yb == y) and (abs(xb - x) < 4):
            if (xb > x):
                action_ideas.append('LEFT')
            if (xb < x):
                action_ideas.append('RIGHT')
            action_ideas.extend(['UP', 'DOWN'])

    for (xb, yb), _ in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    while action_ideas:
        a = action_ideas.pop()
        if a in valid_actions:
            return a
