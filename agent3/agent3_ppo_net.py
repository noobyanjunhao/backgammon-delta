"""
Agent 3 PPO Network: Agent 2's network + Policy Head.

This network extends Agent 2's value network with a policy head:
- Shared ResNet backbone (same as Agent 2)
- Value head (critic): 134 → 64 → 1 (Tanh scaled by 3)
- Policy head (actor): 134 → 64 → 625 (25×25 grid for source-destination pairs)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

# Import Agent 2's network components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define constants (same as Agent 2)
BOARD_LENGTH = 24
CONV_INPUT_CHANNELS = 15
AUX_INPUT_SIZE = 6
FILTERS = 128
NUM_RESIDUAL_BLOCKS = 9


# Policy head outputs 25×25 = 625 logits (source-destination pairs)
ACTION_SPACE_SIZE = 25 * 25


class Agent3PPONet(nn.Module):
    """
    Agent 3 PPO Network with shared backbone:
    - Agent 2's ResNet backbone (24×15 → 24×128 → 128)
    - Value head (critic): 134 → 64 → 1
    - Policy head (actor): 134 → 64 → 625
    """
    
    @nn.compact
    def __call__(self, board_state: jnp.ndarray, aux_features: jnp.ndarray):
        """
        Args:
            board_state: Array of shape (B, 24, 15) [24 points × 15 feature planes]
            aux_features: Array of shape (B, 6) [6 auxiliary features]
        
        Returns:
            (value_pred, policy_logits) tuple:
            - value_pred: (B,) array with values in range [-3, 3]
            - policy_logits: (B, 625) array of logits for 25×25 source-destination pairs
        """
        B = board_state.shape[0]  # Batch size
        
        # Ensure correct input shape
        x = board_state.reshape(B, BOARD_LENGTH, CONV_INPUT_CHANNELS)
        
        # --- SHARED BACKBONE (Agent 2's architecture) ---
        # 1. Initial 1D ConvNet: 24×15 → 24×128
        x = nn.Conv(
            features=FILTERS,
            kernel_size=(7,),
            strides=(1,),
            padding='SAME',
            name='initial_conv_7'
        )(x)
        x = nn.relu(x)
        
        # 2. Stack of 9 ResNet v2 blocks: 24×128 → 24×128
        for i in range(NUM_RESIDUAL_BLOCKS):
            # ResNet block with pre-activation
            input_features = x
            out = nn.LayerNorm(name=f'res_block_{i}_ln_1')(x)
            out = nn.relu(out)
            out = nn.Conv(
                features=FILTERS,
                kernel_size=(3,),
                strides=(1,),
                padding='SAME',
                name=f'res_block_{i}_conv_1'
            )(out)
            out = nn.LayerNorm(name=f'res_block_{i}_ln_2')(out)
            out = nn.relu(out)
            out = nn.Conv(
                features=FILTERS,
                kernel_size=(3,),
                strides=(1,),
                padding='SAME',
                name=f'res_block_{i}_conv_2'
            )(out)
            x = input_features + out
        
        # 3. Global Average Pooling: 24×128 → 128
        x = jnp.mean(x, axis=1)
        
        # 4. Concatenate Auxiliary Features: 128 + 6 = 134
        shared_features = jnp.concatenate([x, aux_features], axis=-1)
        
        # --- CRITIC HEAD (Value) ---
        val = nn.Dense(features=64, name='val_hidden')(shared_features)
        val = nn.relu(val)
        val_out = nn.Dense(features=1, name='val_output')(val)
        value_pred = jnp.tanh(val_out) * 3.0  # Range: [-3, 3]
        
        # --- ACTOR HEAD (Policy) ---
        # 134 → 64 (ReLU) → 625 (linear, no activation)
        pol = nn.Dense(features=64, name='pol_hidden')(shared_features)
        pol = nn.relu(pol)
        policy_logits = nn.Dense(features=ACTION_SPACE_SIZE, name='pol_output')(pol)
        
        return value_pred.squeeze(-1), policy_logits

