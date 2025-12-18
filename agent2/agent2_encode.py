"""
Agent 2 Feature Encoding: Exact specification.

Feature planes (15 total):
  1. Empty (binary, shared)
  2. Blot (count=1) - 2 planes (self, opp)
  3. Made (count=2) - 2 planes (self, opp)
  4. Builder (count=3) - 2 planes (self, opp)
  5. Basic Anchor (count=4) - 2 planes (self, opp)
  6. Deep Anchor (count=5) - 2 planes (self, opp)
  7. Permanent Anchor (count=6) - 2 planes (self, opp)
  8. Overflow (normalized count above 6, out of 9) - 2 planes (self, opp)

Auxiliary features (6 total):
  - Bar Active (binary, 1 if checkers on bar)
  - Bar Scale (normalized count, out of 15)
  - Borne-off (normalized count, out of 15)
  (for each player: self and opponent)
"""

import numpy as np
from backgammon_engine import _to_canonical, B_OFF, W_BAR, B_BAR, W_OFF

BOARD_LENGTH = 24
CONV_INPUT_CHANNELS = 15
AUX_INPUT_SIZE = 6


def encode_agent2(state, player):
    """
    Encode state into Agent 2 feature planes and auxiliary features.
    
    Args:
        state: Board state array (28 elements)
        player: +1 for white, -1 for black
    
    Returns:
        (board, aux) tuple:
        - board: (24, 15) array of feature planes
        - aux: (6,) array of auxiliary features
    """
    s = np.asarray(state, dtype=np.int8)
    
    # Defensive: if state is not a full backgammon board, return zeros
    min_state_size = int(B_OFF) + 1
    if s.ndim != 1 or s.size < min_state_size:
        board = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
        aux = np.zeros((AUX_INPUT_SIZE,), dtype=np.float32)
        return board, aux
    
    # Canonicalize: current player is "positive"
    c = _to_canonical(s, np.int8(player)).astype(np.int16)
    c = np.asarray(c, dtype=np.int16).ravel()
    
    # Ensure we have enough elements
    if c.shape[0] < min_state_size:
        board = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
        aux = np.zeros((AUX_INPUT_SIZE,), dtype=np.float32)
        return board, aux
    
    # Extract 24 points
    pts = c[1:25]  # Points 1-24
    if pts.shape[0] != BOARD_LENGTH:
        if pts.shape[0] < BOARD_LENGTH:
            pts = np.pad(pts, (0, BOARD_LENGTH - pts.shape[0]), mode='constant')
        else:
            pts = pts[:BOARD_LENGTH]
    
    # Clip to valid range
    self_cnt = np.clip(pts, 0, 15).astype(np.int32)
    opp_cnt = np.clip(-pts, 0, 15).astype(np.int32)
    
    # Initialize board: (24, 15)
    board = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    
    # Plane 1: Empty (shared, binary)
    board[:, 0] = (pts == 0).astype(np.float32)
    
    # Planes 2-8: Each has 2 planes (self, opp)
    # Plane 2: Blot (count=1)
    board[:, 1] = (self_cnt == 1).astype(np.float32)  # self
    board[:, 2] = (opp_cnt == 1).astype(np.float32)   # opp
    
    # Plane 3: Made (count=2)
    board[:, 3] = (self_cnt == 2).astype(np.float32)  # self
    board[:, 4] = (opp_cnt == 2).astype(np.float32)   # opp
    
    # Plane 4: Builder (count=3)
    board[:, 5] = (self_cnt == 3).astype(np.float32)  # self
    board[:, 6] = (opp_cnt == 3).astype(np.float32)  # opp
    
    # Plane 5: Basic Anchor (count=4)
    board[:, 7] = (self_cnt == 4).astype(np.float32)  # self
    board[:, 8] = (opp_cnt == 4).astype(np.float32)   # opp
    
    # Plane 6: Deep Anchor (count=5)
    board[:, 9] = (self_cnt == 5).astype(np.float32)  # self
    board[:, 10] = (opp_cnt == 5).astype(np.float32)  # opp
    
    # Plane 7: Permanent Anchor (count=6)
    board[:, 11] = (self_cnt == 6).astype(np.float32)  # self
    board[:, 12] = (opp_cnt == 6).astype(np.float32)    # opp
    
    # Plane 8: Overflow (normalized count above 6, out of 9)
    board[:, 13] = np.maximum(self_cnt - 6, 0).astype(np.float32) / 9.0  # self
    board[:, 14] = np.maximum(opp_cnt - 6, 0).astype(np.float32) / 9.0   # opp
    
    # Auxiliary features (6 total)
    aux = np.zeros((AUX_INPUT_SIZE,), dtype=np.float32)
    
    # Defensive: check bounds before accessing
    self_bar = float(max(c[0] if c.shape[0] > 0 else 0, 0))
    opp_bar = float(max(c[25] if c.shape[0] > 25 else 0, 0))
    self_off = float(max(c[26] if c.shape[0] > 26 else 0, 0))
    opp_off = float(max(c[27] if c.shape[0] > 27 else 0, 0))
    
    # Self player features
    aux[0] = 1.0 if self_bar > 0 else 0.0  # Bar Active
    aux[1] = self_bar / 15.0                 # Bar Scale
    aux[2] = self_off / 15.0                 # Borne-off
    
    # Opponent features
    aux[3] = 1.0 if opp_bar > 0 else 0.0    # Bar Active
    aux[4] = opp_bar / 15.0                  # Bar Scale
    aux[5] = opp_off / 15.0                  # Borne-off
    
    return board, aux

