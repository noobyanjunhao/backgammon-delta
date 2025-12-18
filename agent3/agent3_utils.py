"""
Agent 3 Utilities: Submove indexing and move conversion.
"""

import numpy as np
from backgammon_engine import _get_target_index, NUM_POINTS, W_OFF, B_OFF

MAX_SUBMOVES = 4  # doubles can create up to 4 submoves


def physical_point_to_canonical(point, player):
    """
    Convert a physical point index (0..27) for 'player' into canonical point
    indices (bar=0, points 1..24). Off is handled separately.
    """
    if player == 1:
        # White: bar is 0, points are 1..24 already
        return int(point)
    else:
        # Black: canonical flips using 25 - point
        # Physical black bar is 25 => 25-25=0 (canonical bar)
        return int(NUM_POINTS + 1 - point)  # 25 - point


def physical_target_to_canonical_to_idx(target, player):
    """
    Convert physical target index into canonical 'to_idx' in 0..24 where:
      0 = off
      1..24 = board points
    """
    if player == 1:
        if target == W_OFF:
            return 0
        if 1 <= target <= 24:
            return int(target)
        return 0
    else:
        if target == B_OFF:
            return 0
        if 1 <= target <= 24:
            return int(NUM_POINTS + 1 - target)  # 25 - target
        return 0


def move_to_submove_action_indices(move_seq, player):
    """
    move_seq is a list of (from_point, roll) pairs from engine.
    Convert to list of canonical submove action indices in 0..624.
    Each submove index corresponds to (from_idx, to_idx):
      idx = from_idx * 25 + to_idx
    where from_idx in [0..24], to_idx in [0..24] (0=off, 1..24=points).
    """
    idxs = []
    for (from_point, roll) in move_seq:
        from_point = int(from_point)
        roll = int(roll)
        target = int(_get_target_index(np.int8(from_point), np.int8(roll), np.int8(player)))

        from_idx = physical_point_to_canonical(from_point, player)  # 0..24
        to_idx = physical_target_to_canonical_to_idx(target, player)  # 0..24

        # clamp safety
        from_idx = max(0, min(24, from_idx))
        to_idx = max(0, min(24, to_idx))

        idxs.append(from_idx * 25 + to_idx)

    # pad to MAX_SUBMOVES with -1
    out = np.full((MAX_SUBMOVES,), -1, dtype=np.int32)
    for i in range(min(len(idxs), MAX_SUBMOVES)):
        out[i] = np.int32(idxs[i])
    return out


def submove_idx_to_from_to(submove_idx):
    """
    Convert submove index (0..624) back to (from_idx, to_idx).
    """
    from_idx = submove_idx // 25
    to_idx = submove_idx % 25
    return from_idx, to_idx

