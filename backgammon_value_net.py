import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

# --- Hyperparameters ---
INPUT_CHANNELS_PER_PLAYER = 5
BOARD_LENGTH = 24
CONV_INPUT_CHANNELS = 9
AUX_INPUT_SIZE = 6 # 5 Aux + 1 Cube
FILTERS = 128
NUM_RESIDUAL_BLOCKS = 6

# --- 1D ResNet V2 Residual Block (Pre-activation) ---
class ResidualBlockV2(nn.Module):
    """
    Implements a 1D Residual Block with Pre-activation (V2) structure.
    LayerNorm and ReLU are applied *before* the convolutional layers.
    """
    channels: int
    kernel_size: int = 3
    
    # We use a custom Name for the BatchNorm layer to avoid sharing parameters 
    # across different ResBlocks if we instantiate them within a list/loop.
    @nn.compact
    def __call__(self, x):
        input_features = x
        
        # --- Skip connection starts here (Input x is preserved) ---
        
        # 1. Pre-activation (BN + ReLU)
        # nn.BatchNorm needs to be defined outside the __call__ method for standard Flax use,
        # but nn.compact allows for inline definition using nn.module_cls
        out = nn.LayerNorm(name='bn_1')(x)
        out = nn.relu(out)

        # 2. First 1D Conv (Kernel size 3)
        out = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size,),
            strides=(1,),
            padding='SAME'
        )(out)

        # 3. Second Pre-activation (BN + ReLU)
        out = nn.LayerNorm(name='bn_2')(out)
        out = nn.relu(out)

        # 4. Second 1D Conv (Kernel size 3, final features)
        out = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size,),
            strides=(1,),
            padding='SAME'
        )(out)
        
        # 5. Skip Connection: Add the original input (x) to the output (out)
        return input_features + out

# --- Main Value Network Module ---
class BackgammonValueNet(nn.Module):
    """
    Main Flax module for the 1D Residual ConvNet Value Function.
    """
    @nn.compact
    def __call__(self, board_state: jnp.ndarray, aux_features: jnp.ndarray):
        """
        Args:
            board_state: Array of shape (B, 216) [Flattened 24x9 board].
            aux_features: Array of shape (B, 6) [Global game features].
        """
        B = board_state.shape[0] # Batch size
        
        # 1. Reshape Input: (B, 216) -> (B, L=24, C=9)
        # Numba/Haiku used reshape; Flax needs a slightly different approach 
        # but since input is flat, we manually define the dimensions.
        x = board_state.reshape(B, BOARD_LENGTH, CONV_INPUT_CHANNELS)
        
        # 2. Initial Feature Expansion (Kernel 7: The movement window)
        # Expand channels from 9 to FILTERS (128)
        x = nn.Conv(
            features=FILTERS,
            kernel_size=(7,), # CRUCIAL: Capture max 6-pip move radius + destination
            strides=(1,),
            padding='SAME',
            name='initial_conv_7'
        )(x)
        x = nn.relu(x)

        # 3. Stack Residual Blocks (Kernel 3 for feature synthesis)
        for i in range(NUM_RESIDUAL_BLOCKS):
            # Pass the 128 channels to the V2 Block
            x = ResidualBlockV2(
                channels=FILTERS, 
                kernel_size=3,
                name=f'res_block_{i}'
            )(x)

        # 4. Global Average Pooling (Spatial Collapse)
        # Collapse the spatial dimension (L=24) to get a (B, 128) feature vector
        # [0, 1, 2] -> axis=1 is the length dimension
        x = jnp.mean(x, axis=1)

        # 5. Concatenate Auxiliary Features
        # New shape: (B, 128 + 6) = (B, 134)
        x = jnp.concatenate([x, aux_features], axis=-1)

        # 6. Final Dense Block (Integration and Prediction)
        
        # Dense Hidden Layer (Integration)
        x = nn.Dense(features=64, name='dense_hidden')(x)
        x = nn.relu(x)

        # Final Output Layer (Equity Prediction)
        v_raw = nn.Dense(features=1, name='value_output')(x)
        
        # Use Tanh activation to constrain the output to [-1, 1]
        # (This is scaled externally to match the true game score range, e.g., [-3, +3])
        equity_estimate = jnp.tanh(v_raw)

        return equity_estimate
