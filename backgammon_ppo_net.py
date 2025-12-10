import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

# --- Hyperparameters ---
INPUT_CHANNELS_PER_PLAYER = 5
BOARD_LENGTH = 24
CONV_INPUT_CHANNELS = 15
AUX_INPUT_SIZE = 6 
FILTERS = 128
NUM_RESIDUAL_BLOCKS = 9 
MAX_SCORE_SCALE = 3.0 
ACTION_SPACE_SIZE = 25 * 25 # 625 Canonical Actions

# ... (ResidualBlockV2 class identical to backgammon_value_net) ...
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

class BackgammonPPONet(nn.Module):
    """
    PPO Model with Shared Backbone and Dice-Free Policy Head.
    """
    @nn.compact
    def __call__(self, board_state: jnp.ndarray, aux_features: jnp.ndarray):
        
        B = board_state.shape[0]
        
        # --- SHARED BACKBONE ---
        x = board_state.reshape(B, BOARD_LENGTH, CONV_INPUT_CHANNELS)
        x = nn.Conv(features=FILTERS, kernel_size=(7,), strides=(1,), padding='SAME', name='initial_conv_7')(x)
        x = nn.relu(x)

        for i in range(NUM_RESIDUAL_BLOCKS):
            x = ResidualBlockV2(channels=FILTERS, kernel_size=3, name=f'res_block_{i}')(x)

        x = jnp.mean(x, axis=1) # Global Pooling -> (B, 128)
        
        # Concatenate Aux Features -> (B, 134)
        shared_features = jnp.concatenate([x, aux_features], axis=-1)

        # --- CRITIC HEAD (Value) ---
        # Estimates Afterstate Equity (No Dice Dependency)
        val = nn.Dense(features=64, name='val_hidden')(shared_features)
        val = nn.relu(val)
        val_out = nn.Dense(features=1, name='val_output')(val)
        value_pred = jnp.tanh(val_out) * MAX_SCORE_SCALE

        # --- ACTOR HEAD (Policy) ---
        # Estimates Strategic Move Desirability (No Dice Dependency)
        # Outputs 625 canonical move logits
        pol = nn.Dense(features=64, name='pol_hidden')(shared_features)
        pol = nn.relu(pol)
        policy_logits = nn.Dense(features=ACTION_SPACE_SIZE, name='pol_output')(pol)

        return value_pred, policy_logits
