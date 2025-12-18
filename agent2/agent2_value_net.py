"""
Agent 2 Value Network: 1D ConvNet with ResNet v2 blocks.

This network implements the exact architecture specified for Agent 2:
- 24 inputs × 15 channels (feature planes)
- Initial 1D ConvNet (kernel=7, stride=1) → 24×128
- Stack of 9 ResNet v2 blocks (kernel=3, stride=1)
- Global average pooling → 128
- Dense layers: 134 → 64 → 1 (Tanh scaled by 3)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

# --- Hyperparameters ---
BOARD_LENGTH = 24
CONV_INPUT_CHANNELS = 15  # Feature planes: 1 empty + 7×2 (per player) = 15
AUX_INPUT_SIZE = 6  # Bar Active, Bar Scale, Borne-off for each player
FILTERS = 128
NUM_RESIDUAL_BLOCKS = 9


# --- 1D ResNet V2 Residual Block (Pre-activation) ---
class ResidualBlockV2(nn.Module):
    """
    Implements a 1D Residual Block with Pre-activation (V2) structure.
    
    Formula: x_out = x_in + Conv(ReLU(LayerNorm(Conv(ReLU(LayerNorm(x_in))))))
    
    The skip connection bypasses the pre-activation layers.
    """
    channels: int
    kernel_size: int = 3
    
    @nn.compact
    def __call__(self, x):
        input_features = x  # Save for skip connection
        
        # First pre-activation: LayerNorm + ReLU
        out = nn.LayerNorm(name='ln_1')(x)
        out = nn.relu(out)
        
        # First 1D Conv (Kernel size 3)
        out = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size,),
            strides=(1,),
            padding='SAME',
            name='conv_1'
        )(out)
        
        # Second pre-activation: LayerNorm + ReLU
        out = nn.LayerNorm(name='ln_2')(out)
        out = nn.relu(out)
        
        # Second 1D Conv (Kernel size 3)
        out = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size,),
            strides=(1,),
            padding='SAME',
            name='conv_2'
        )(out)
        
        # Skip connection: add original input (bypasses pre-activation)
        return input_features + out


# --- Main Agent 2 Value Network Module ---
class Agent2ValueNet(nn.Module):
    """
    Agent 2 Value Network with exact specification:
    - 1D ConvNet (24×15 → 24×128, kernel=7)
    - 9 ResNet v2 blocks (24×128 → 24×128, kernel=3)
    - Global average pooling (24×128 → 128)
    - Dense layers (134 → 64 → 1, Tanh scaled by 3)
    """
    
    @nn.compact
    def __call__(self, board_state: jnp.ndarray, aux_features: jnp.ndarray):
        """
        Args:
            board_state: Array of shape (B, 24, 15) [24 points × 15 feature planes]
            aux_features: Array of shape (B, 6) [6 auxiliary features]
        
        Returns:
            Array of shape (B, 1) with values in range [-3, 3]
        """
        B = board_state.shape[0]  # Batch size
        
        # Ensure correct input shape
        x = board_state.reshape(B, BOARD_LENGTH, CONV_INPUT_CHANNELS)
        
        # 1. Initial 1D ConvNet: 24×15 → 24×128
        # Kernel size 7 captures full 6-pip window
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
            x = ResidualBlockV2(
                channels=FILTERS,
                kernel_size=3,
                name=f'res_block_{i}'
            )(x)
        
        # 3. Global Average Pooling: 24×128 → 128
        # Collapse spatial dimension (axis=1)
        x = jnp.mean(x, axis=1)
        
        # 4. Concatenate Auxiliary Features: 128 + 6 = 134
        x = jnp.concatenate([x, aux_features], axis=-1)
        
        # 5. Final Dense Layers: 134 → 64 → 1
        # Hidden layer with ReLU
        x = nn.Dense(features=64, name='dense_hidden')(x)
        x = nn.relu(x)
        
        # Output layer with Tanh, scaled by 3
        v_raw = nn.Dense(features=1, name='value_output')(x)
        equity_estimate = jnp.tanh(v_raw) * 3.0  # Range: [-3, 3]
        
        return equity_estimate

