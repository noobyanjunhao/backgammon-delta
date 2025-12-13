import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax
import numpy as np
import numba
import orbax
from flax import struct # Use for defining immutable state

LEARNING_RATE = 1e-4
LAMBDA = 0.7 # The trace parameter
GAMMA = 0.99 # The discount factor
MAX_BATCH_SIZE = 2*2*2*8192
MIN_BATCH_SIZE = 64

# CONV_INPUT_CHANNELS = 15
# AUX_INPUT_SIZE = 6
# BOARD_LENGTH = 24

# INITIALIZATION. Process the model definition in backgammon_value.net.py:

from backgammon_value_net import *

model = BackgammonValueNet()

# Initialize model parameters using randomly generated weights
# (generated deterministically by jax from a provided random seed, by
# calling model.init(). We have to provide it with dummy inputs, because
# jax permits model definitions to be concise without pinning down all
# vector shapes. So model.init() infers them from the provided inputs.
# Here we generate a batch of 32 inputs with zeros.

dummy_planes= jnp.zeros( (32, BOARD_LENGTH, CONV_INPUT_CHANNELS) )
dummy_aux_features = jnp.zeros( (32, AUX_INPUT_SIZE) )
rng_key = jax.random.key(0)

init_variables = model.init(
    rng_key,
    board_state = dummy_planes,
    aux_features = dummy_aux_features
)

# model.init() returns a jax "pytree" dictionary, with the initial
# (random) weights for the network stored under the key 'params'

init_params = init_variables['params']

# Instead of being a single big numpy vector, this weight vector is a
# nested jax "pytree" dictionary structure of labeled vectors for the
# different components of the network:
#
# {'initial_conv_7': {'kernel': Array([[[-0.16254435,  0.00062753,  0.05183868, ...,  0.16529857,
#           -0.09350016,  0.05186949],
#          [-0.19029863,  0.13044679, -0.04520517, ...,  0.06310008,
# ...

# EVALUATION

# To evaluate the network on a batch of input vectors, say a batch of
# 10 vectors, all of whose entries are equal to 1 (the first tensor
# index specifies the vector within the batch)

test_planes = jnp.ones( (10, BOARD_LENGTH, CONV_INPUT_CHANNELS) )
test_aux_features = jnp.ones( (10, AUX_INPUT_SIZE) )

# call model.apply()

predicted_values = model.apply(
    {'params': init_params},     # 1. The PyTree of weights
    test_planes, test_aux_features  # 2. The input data
)

# model.apply() returns a batch of outputs, for our value network,
# this is a jax.Array of shape (10, 1)

# When you call model.apply(), jax compiles optimized gpu code for
# your particular gpu/tpu/cpu setup to evaluate the network
# specifically on a batch of 10 input vectors, and then caches this
# compiled code for future calls. So the first call is slow, but
# future calls are fast.

# If we changed to

test_planes= jnp.ones( (20, BOARD_LENGTH, CONV_INPUT_CHANNELS) )
test_aux_features = jnp.ones( (20, AUX_INPUT_SIZE) )

# and then call model.apply() a second time

predicted_values = model.apply(
    {'params': init_params},     # 1. The PyTree of weights
    test_planes, test_aux_features  # 2. The input data
)

# jax compiles new optimized gpu code specifically for a batch of *20*
# input vectors, and caches this compiled code for future calls. So
# each time you call model.apply() on a new size of batch, it incurs a
# compliation penalty.

# Since for us the number of states required to be evaluated for 2-ply
# search is dynamic, a naive implementation will end up asking jax to
# compile many different versions of the evaluation code.

# It is recommended that you write code to use only a few different
# batch sizes, say powers of two, and either break any given batch
# into subbatches, or else pad your batch with zeros to the nearest
# power of two, or some combination of these strategies.

# TRAINING

# Recall that gradient descent minimizes a loss function l(w) by
# iteratively computing a sequence of weights w_1, w_2, w_3, ... by
# setting
#   w_{t+1} = w_t - alpha gradient(l(w_t))
# where alpha is the learning rate.

# And that the idea of stochastic gradient descent is to estimate l(w)
# by some random variable L(w) having
# E[gradient(L(w))] = gradient(l(w))
# and to compute a sequence of weights w_1, w_2, w_3,... by setting
#   w_{t+1} = w_t - alpha gradient(L(w_t)).

# Instead of using a constant learning rate alpha, a key to the deep
# learning revolution is to use the Adam optimizer to adapt alpha on a
# per-component basis in a sophisticated way
# https://arxiv.org/abs/1412.6980

# (Note that we cannot implement True Online TD(lambda) using the Adam
# optimizer, since we would need to update the trace vector with the
# same adapted alpha as we are using to train the network. Or rather,
# to figure out how to do so is a subject of ongoing research.)

optimizer = optax.adam(learning_rate=LEARNING_RATE)

# The optimizer itself maintains a state (to track how stable the
# different components are)
opt_state = optimizer.init(init_params)

# To train the network, you define a loss function L (here is a
# simplified version where you are passing in a vector of targets

def loss_fn(params, state_planes, aux_features, reward_targets):
    # 1. Apply the model with current parameters
    # The agent's model needs to be accessible in this scope
    V_pred = model.apply({'params': params}, state_planes, aux_features)
    
    # 2. Calculate the Mean Squared Error (MSE) loss
    td_error = reward_targets - V_pred
    loss = jnp.mean(jnp.square(td_error))
    
    return loss

# This is our stochastic approximation L to the ideal loss function
# (evaluated on batches of samples), and we ask jax to minimize this
# function through stochastic gradient descent. The basic flow is to
# call jax.value_and_grad applied to whatever loss function we are
# interested in minimizing, in order to have jax compute the error and
# the gradient simultaneously (in an optimized, distributed way across
# the gpu), and then pass this information to the optimizer to perform
# the update to the weights.

# An important technical point for using jax distinguishing it from
# other deep learning frameworks: Even though we interact with jax
# from Python, which has mutable data structures, all values in jax
# (even tensors) are immutable (which enables it to optimize its
# computational graph across the gpu's better). So jax never updates
# weight vectors in place. Instead it returns NEW weight vectors, and
# a NEW optimizer state.

@jax.jit
def train_step(params, opt_state, state_planes, aux_features, reward_targets):
    # 1. Compute Loss AND Gradients (Efficient single pass)
    # The output is a tuple: (scalar_loss, gradients_pytree)
    (loss_value, gradients_pytree) = jax.value_and_grad(loss_fn)(
        params, 
        state_planes,
        aux_features,
        reward_targets
    )
    
    # 2. Compute Updates (Step 1 of Optax)
    #
    # The optimizer uses its internal state (momentum/variance) to calculate weight adjustments
    # fancy version of weight update w_{t+1} = w_t + alpha gradient( L(w_t) )

    updates, new_opt_state = optimizer.update(
        gradients_pytree, 
        opt_state, 
        params
    )
    
    # 3. Apply Updates (Step 2 of Optax)
    # The arithmetic step: new_params = params + updates
    new_params = optax.apply_updates(params, updates)
    
    # Return the new state and the loss value (for logging)
    return new_params, new_opt_state, loss_value

# Let's do a training update with reward targets

rand_rewards = np.random.random(20)

new_params, new_opt_state, loss = train_step( init_params, opt_state, test_planes, test_aux_features, rand_rewards)

# Now after you spend hours training a network, you want to save it.
# (Better yet, regularly save "checkpoints" along the way.)
# The best tool for this for jax is orbax.

import orbax.checkpoint as ocp
import os
import pathlib

# Orbax insists that this path be absolute
CHECKPOINT_ROOT = '/tmp/my_checkpoints'
path = pathlib.Path(CHECKPOINT_ROOT)
os.makedirs(path, exist_ok=True)

# path = ocp.test_utils.erase_and_create_empty('/tmp/my_checkpoints/')

checkpointer = ocp.StandardCheckpointer()

# By default Orbax will not overwrite existing data, force=True tells it to overwrite
checkpointer.save(path / 'new_params', new_params, force=True)

restored_params = checkpointer.restore(
    path / 'new_params/' )

print(restored_params)

# Orbax writes are asynchronous, close to make sure all writing finishes
checkpointer.close() 


# To create a vector of the "same shape" as params (a nested pytree
# with the the same labels and vector shapes) take params and "map" a
# function onto each vector within it with the function tree_map
trace = tree_map(lambda p: jnp.zeros_like(p), init_params)

# To add two such vectors, you need to do a
trace1 = trace # silly example
trace2 = trace # silly example

new_trace = tree_map(
    lambda z, g: GAMMA * LAMBDA * z + g, 
    trace1, 
    trace2
)
