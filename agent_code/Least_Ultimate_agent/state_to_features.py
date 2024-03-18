import numpy as np

def state_to_features(game_state: dict, current_explosion_map, old_explosion_map) -> np.array:
    if game_state is None:
        return None

    pad_r = 5  # How far the padding goes
    pad_v = -1  # The padding value

    _, _, _, (x_me, y_me) = game_state["self"]
    field_shape = np.pad(game_state["field"], (pad_r, pad_r), constant_values=(pad_v, pad_v)).shape
    x_me_pad = x_me + pad_r
    y_me_pad = y_me + pad_r

    # Create a hybrid matrix with field shape x vector of size 6 to encode field state
    hybrid_matrix = np.zeros((6,) + field_shape, dtype=np.double)

    # Encode other players
    for _, _, _, (x, y) in game_state["others"]:
        hybrid_matrix[0, x + pad_r, y + pad_r] = 1

    # Encode bombs
    for (x, y), timer in game_state["bombs"]:
        hybrid_matrix[1, x + pad_r, y + pad_r] = 1 / (1 + timer)

    # Encode crates
    hybrid_matrix[2, :, :] = np.pad(np.where(game_state["field"] == 1, 1, 0), (pad_r, pad_r), constant_values=(pad_v, pad_v))

    # Encode walls
    hybrid_matrix[3, :, :] = np.pad(np.where(game_state["field"] == -1, -5, 0), (pad_r, pad_r), constant_values=(pad_v, pad_v))

    # Encode coins
    for (x, y) in game_state["coins"]:
        hybrid_matrix[4, x + pad_r, y + pad_r] = 10

    # Encode explosions
    exp = game_state["explosion_map"] + current_explosion_map + old_explosion_map
    hybrid_matrix[5, :, :] = np.pad(exp, (pad_r, pad_r), constant_values=(pad_v, pad_v))

    # Extract the relevant part of the hybrid matrix around the player
    reduced_hybrid = hybrid_matrix[:, x_me_pad - pad_r:x_me_pad + pad_r, y_me_pad - pad_r:y_me_pad + pad_r]

    return reduced_hybrid
