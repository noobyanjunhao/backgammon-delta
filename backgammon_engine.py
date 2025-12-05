import numpy as np
import numba
from numba import njit, types, prange
from numba.typed import List, Dict

# Backgammon code for CS McTague's Reinforcement Learning course at
# Boston College, see https://cs.bc.edu/mctague/t/2025/rl/fproj/

# Type Aliases
int8 = types.int8
Player = int8
Dice = np.ndarray[int8] # equal dice should be interpreted correctly
State = types.Array(types.int8, ndim=1, layout='C')
Action = types.ListType(types.UniTuple(types.int8,2)) # 4 submoves
StateTuple = types.UniTuple(types.int8, 28)

TwoInt8Tuple = types.UniTuple(types.int8, 2)
ThreeInt8Tuple = types.UniTuple(types.int8, 3)
FourInt8Tuple = types.UniTuple(types.int8, 4)

# Constants for Indices
NUM_DICE = 2
NUM_SORTED_ROLLS = 21 # 21 sorted rolls
NUM_POINTS = int8(24)
NUM_CHECKERS = int8(15)
HOME_BOARD_SIZE = int8(6)
W_BAR, B_BAR, W_OFF, B_OFF = int8(0), int8(NUM_POINTS+1), int8(NUM_POINTS+2), int8(NUM_POINTS+3)
STATE_SIZE = 28

@njit
def _new_game():
    # Generate new game. Returns board in standard starting setup and
    # performs initial dice roll to determine which player starts
    # first. Returns: Starting player, their dice roll, and board
    # state.

    # Initial standard setup (P1, P2...P24). Indices 1-24 correspond to points.
    state = np.array( [
        0, # W_BAR (Index 0)
        2, 0, 0, 0, 0, -5,  # P1 to P6
        0, -3, 0, 0, 0, 5,  # P7 to P12
        -5, 0, 0, 0, 3, 0,  # P13 to P18
        5, 0, 0, 0, 0, -2,  # P19 to P24
        0, # B_BAR (Index 25)
        0, 0  # W_OFF (Index 26), B_OFF (Index 27)
    ], dtype=int8 )

    dice = _roll_dice()
    while( dice[0] == dice[1] ):
        dice = _roll_dice()

    player = int8( 2*np.random.randint(0,2) - 1 )

    return player, dice, state

@njit
def _roll_dice():
    # Generates a roll of two standard dice, sorted.
    return np.sort( np.random.randint(1,7,size=2).astype(np.int8) )[::-1]

@njit
def _can_bear_off(state, player):
    # Checks if all a players's pieces are in their home quadrant (and
    # none are on the bar.

    bar_index = W_BAR if player == 1 else B_BAR
    if state[bar_index] * player > 0:
        return False

    # Check for pieces outside the home board
    # White's home board: P19-P24 (Indices 19-24)
    if player == 1:
        # Check P1 to P18 (Indices 1 to 18)
        for i in range(1, NUM_POINTS - HOME_BOARD_SIZE + 1):
            if state[i] > 0: # White checker found outside home
                return False
    # Black's home board: P1-P6 (Indices 1-6)
    else:
        # Check P7 to P24 (Indices 7 to 24)
        for i in range( HOME_BOARD_SIZE+1, NUM_POINTS + 1 ):
            if state[i] < 0: # Black checker found outside home
                return False

    return True

@njit
def _get_target_index(start_index, roll, player):
    # Returns the target index when applying a roll to a checker in
    # position start_index.
    target = int8(start_index + roll * player)
    if 1 <= target <= NUM_POINTS:
        return target
    if target <= 0:
        return W_OFF
    return B_OFF

@njit
def _is_move_legal(state, player, from_point, to_point):
    # Checks if a move (from, to) is instantaneously legal. Assumes
    # from_point has a checker and to_point is reached by a valid die
    # roll.

    # Rule 1: Cannot move from an empty point
    if state[from_point] * player <= 0:
        return False

    # Rule 2: Cannot move if pieces are on the bar unless the move starts at the bar
    bar_index = W_BAR if player == 1 else B_BAR
    if state[bar_index] * player > 0 and from_point != bar_index:
        return False

    # Rule 3: Target point must not be blocked by the opponent
    if 1 <= to_point <= NUM_POINTS:
        target_checkers = state[to_point] * player
        if target_checkers <= -2:
            return False

    # Rule 4: If bearing off, must satisfy the Furthest Back (Clearance) Rule.
    # Check if there is any piece further back than the one being moved.
    if to_point == W_OFF or to_point == B_OFF:
        off_target = W_OFF if player == 1 else B_OFF
        if to_point != off_target:
            return False
        if _can_bear_off(state, player):
            return False
        if player == 1:
            for i in range( NUM_POINTS - HOME_BOARD_SIZE + 1,
                            from_point):
                if state[i] > 0: # Found a piece further back
                    return False 
        else:
            for i in range(from_point - 1, 0, -1):
                if state[i] < 0: # Found a piece further back
                    return False 

    return True

@njit
def _apply_sub_move(state: State, player: int, from_point: int, to_point: int):
    # Applies a single (from, to) move to a temporary state, including
    # hitting/bearing off. Returns the resulting state if legal, or
    # None if the move is invalid.

    if not _is_move_legal(state, player, from_point, to_point):
        return None

    next_state = state.copy()

    # Remove checker from source
    next_state[from_point] -= player

    # If bearing off
    if to_point == W_OFF or to_point == B_OFF:
        next_state[to_point] += player

    # If not bearing off
    elif 1 <= to_point <= NUM_POINTS:
        # Is there an opponent blot? Move it to bar
        if next_state[to_point] == -player:
            opponent_bar_index = B_BAR if player == 1 else W_BAR
            next_state[opponent_bar_index] -= player
            next_state[to_point] = player
        else:
            next_state[to_point] += player
    else:
        return None

    return next_state

@njit
def _apply_move(state, player, move_sequence):
    # Applies a full move (a sequence of submoves, ie an Action) to
    # the state.
    current_state = state.copy()
    
    for from_point, roll in move_sequence:
        to_point = _get_target_index( from_point, roll, player)
        result_state = _apply_sub_move(current_state, player, from_point, to_point)
        # This should not fail if the sequence came from get_valid_moves
        if result_state is None:
            raise ValueError("Attempted to apply an invalid move sequence step.")
        current_state = result_state
    return current_state

@njit
def _reward(state, player):
    if state[W_OFF] == NUM_CHECKERS:
        # white has won
        if state[B_OFF] < 0:
            # if the losing player has borne off at least one checker,
            # he loses only one point
            return player
        else:
            # if the loser has not borne off any of his checkers, he is
            # gammoned and loses two points
            #
            # if the loser has not borne off any of his checkers and still has
            # a checker on the bar or in the winner's home board, he is
            # backgammoned and loses three points
            if state[B_BAR] < 0:
                return 3*player
            for p in range(1,HOME_BOARD_SIZE+1):
                if state[p] < 0:
                    return 3*player
            return 2*player
    if state[B_OFF] == NUM_CHECKERS:
        # black has won
        if state[W_OFF] > 0:
            return -player
        else:
            if state[W_BAR] > 0:
                return -3*player
            for p in range(3*HOME_BOARD_SIZE+1,NUM_POINTS+1):
                if state[p] > 0:
                    return -3*player
            return -2*player
    return 0

@njit
def _actions(state, current_player, dice):
    # Generates an exhaustive list of all legal move sequences for the
    # current player for the given dice. Returns a numba list of moves
    # (Action) and a parallel list of afterstates.

    all_moves = List.empty_list( Action )
    all_afterstates = List.empty_list( State )
    
    current_move = List.empty_list( TwoInt8Tuple )

    dice_list = List(dice)
    if( dice[0] == dice[1] ):
        dice_list = List([dice[0], dice[0], dice[0], dice[0]])

    _find_moves_recursive(state, current_player,
                          dice_list, current_move, all_moves, all_afterstates)

    if len(all_moves) == 0:
        # If no moves were possible at all (Forced Pass)
        return all_moves, all_afterstates
    
    # Player must use maximum number of dice possible

    max_dice_used = 0
    for move in all_moves:
        l = len(move)
        if l > max_dice_used:
            max_dice_used = l
            if max_dice_used == len(dice_list):
                break

    max_len_moves = List.empty_list( Action )
    max_len_afterstates = List.empty_list( State )

    for i in range(len(all_moves)):
        if len(all_moves[i]) == max_dice_used:
            max_len_moves.append(all_moves[i])
            max_len_afterstates.append(all_afterstates[i])
            
    if max_dice_used == len(dice_list):
        return max_len_moves, max_len_afterstates

    # Player must use maximum number of pips possible
    
    total_pips = sum(dice_list)
    max_pips_used = 0
    for move in max_len_moves:
        s = sum( [ r for f,r in move ] )
        if s > max_pips_used:
            max_pips_used = s
            if max_pips_used == total_pips:
                break

    max_pip_moves = List.empty_list( Action )
    max_pip_afterstates = List.empty_list( State )


    for i in range(len(max_len_moves)):
        s = sum( [ r for f,r in max_len_moves[i] ] )
        if s == max_pips_used:
            max_pip_moves.append( max_len_moves[i] )
            max_pip_afterstates.append( max_len_afterstates[i] )

    return max_pip_moves, max_pip_afterstates

@njit
def _state_to_tuple(a):
    # if anyone can find a better way, which works in numba nopython,
    # lmk! not only is this ugly, it generates a
    # "NumbaTypeSafetyWarning: unsafe cast from UniTuple(int64 x 28)
    # to UniTuple(int8 x 28). Precision may be lost."
    return (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
            a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16],
            a[17], a[18], a[19], a[20], a[21], a[22], a[23], a[24],
            a[25], a[26], a[27])

@njit
def _unique_afterstates(moves, afterstates):
    # takes parallel arrays of moves and afterstates, and constructs a
    # dictionary whose keys are afterstates and whose values are
    # actions (just one) leading to that afterstate. A list of unique
    # afterstates can thus be extracted by enumerating its keys, for
    # efficient 2-ply search, and a lookup of an afterstate yields a
    # move leading to that afterstate. Because moves are arrays, which
    # are mutable, they cannot be used as keys for a dictionary, so we
    # have to convert the afterstates to tuples, which are immutable.
    # This is surprisingly difficult to do in a way which satisfies
    # the numba type checker.
    afterstate_dict = Dict.empty( StateTuple, Action )

    for i in range(len(moves)):
        afterstate_dict[ _state_to_tuple(afterstates[i]) ] = moves[i]

    return afterstate_dict

@njit
def _find_moves_recursive(state, player, remaining_dice, current_move, all_moves, all_afterstates):
    # Recursive helper to find all possible move sequences given
    # remaining dice. Appends valid moves pairs to the provided numba
    # list all_moves, and the resulting afterstate in the parallel list
    # all_afterstates. Returns nothing.

    # Base Case 1: All dice used
    if len(remaining_dice) == 0:
        all_moves.append( current_move )
        all_afterstates.append( state )
        return

    # Optimization: Only consider unique dice values for iteration
    unique_dice = sorted(list(set(remaining_dice)), reverse=True)

    # Try to use the largest unique die first
    for roll in unique_dice:

        # --- Identify all possible moves for this 'roll' ---
        possible_moves = List.empty_list(ThreeInt8Tuple)

        # If on bar, only consider moves from the bar
        bar_index = W_BAR if player == 1 else B_BAR
        if state[bar_index] * player > 0:
            from_point = bar_index
            target_point = _get_target_index(from_point, roll, player)

            if target_point >= 0:
                if _is_move_legal(state, player, from_point, target_point):
                    possible_moves.append((from_point, target_point, roll))

        # If not on bar
        else:
            for from_point in range(1, NUM_POINTS + 1):
                from_point = int8(from_point)
                if state[from_point] * player > 0:
                    target_point = _get_target_index(from_point, roll, player)

                    if _is_move_legal(state, player, from_point, target_point):
                        possible_moves.append((from_point, target_point, roll))

        # --- Recurse on valid moves found ---
        for from_point, to_point, roll in possible_moves:

            # 1. Apply the move
            next_state = _apply_sub_move(state, player, from_point, to_point)

            # 2. Update remaining dice (remove the die that was just used)
            new_dice_list = list(remaining_dice)
            try:
                new_dice_list.remove(roll)
            except Exception:
                # Should not happen if logic is correct
                continue

            # 3. Recurse with the new state and remaining dice
            new_move = List(current_move)
            new_move.append( (from_point, int8(roll)) )
            _find_moves_recursive( next_state, player, new_dice_list,
                                       new_move, all_moves, all_afterstates)

    if current_move:
        all_moves.append( current_move )
        all_afterstates.append( state )

@njit
def _to_canonical(state, player):
    # Transforms the physical board state into the canonical view (the
    # current player's perspective, where they move from P1 to P24).

    if player == 1:
        # Already in canonical form (White's perspective)
        return state.copy()

    canonical_state = state.copy()

    canonical_state[W_OFF] = -state[B_OFF]
    canonical_state[B_OFF] = -state[W_OFF]

    for i in range(0, NUM_POINTS + 2):
        canonical_state[i] = -state[ NUM_POINTS + 1 - i ]

    return canonical_state

@njit(parallel=True)
def _vectorized_new_game(num_games):
    player_vector = np.empty( num_games, dtype=int8 )
    dice_vector = np.empty( (num_games, NUM_DICE), dtype=int8) 
    state_vector = np.empty( (num_games,STATE_SIZE), dtype=int8 )

    for i in prange(num_games):
        player_vector[i], dice_vector[i], state_vector[i] = _new_game()

    return state_vector, player_vector, dice_vector

@njit(parallel=True)
def _vectorized_roll_dice(num_games):
    rolls = np.random.randint(1,7,size=(num_games,2)).astype(np.int8)
    sorted_dice = np.empty_like(rolls)

    for i in prange(num_games):
        d1 = rolls[i,0]
        d2 = rolls[i,1]
        if d1 >= d2:
            sorted_dice[i,0] = d1
            sorted_dice[i,1] = d2
        else:
            sorted_dice[i,0] = d2
            sorted_dice[i,1] = d1
    return sorted_dice

@njit(parallel=True)
def _vectorized_actions_parallel(state_vector, player_vector, dice_vector):
    actions = List()
    afterstates = List()

    thread_counts = [0] * 24
    
    for i in range( len(state_vector) ):
        actions.append(List.empty_list(Action))
        afterstates.append(List.empty_list(State))
    
    for i in prange( len(state_vector) ):
        actions[i], afterstates[i] = _actions( state_vector[i],
                                               player_vector[i],
                                               dice_vector[i] )

    return actions, afterstates

@njit(parallel=True)
def _vectorized_apply_move(state_vector, player_vector, dice_vector, move_sequence_vector):

    new_state_vector = np.empty( len(state_vector), dtype=State )
    
    for i in prange( len(state_vector) ):
        new_state_vector[i] = _apply_move( state_vector[i],
                                           player_vector[i],
                                           dice_vector[i],
                                           move_sequence_vector[i] )
    return new_state_vector

@njit
def _collect_search_data( state, player, dice ):
    # returns an array of states at the leaves of the 2-ply search
    # tree for batch evaluation by the (neural network) value function

    player_moves, player_afterstates = _actions( state, player, dice )
    afterstates_dict = _unique_afterstates(player_moves, player_afterstates)
    states_buffer = List.empty_list(state)
    offsets = np.empty( (len(afterstates_dict), NUM_SORTED_ROLLS), np.int64)
    i = 0
    m = 0
    for opponent_state in afterstates_dict:
        # convert from tuple back to array
        opponent_state = np.array( opponent_state, dtype=int8 )
        d=0
        for r1 in range(1,7):
            for r2 in range(1,r1+1):
                offsets[m,d] = i
                d = d + 1

                opponent_dice = np.array([r1,r2], dtype=int8)
                opponent_moves, opponent_afterstates = _actions( opponent_state,
                                                                 -player,
                                                                 opponent_dice )
                opponent_afterstates = _unique_afterstates( opponent_moves,
                                                            opponent_afterstates )
                
                for opponent_afterstate in opponent_afterstates:
                    opponent_afterstate = np.array( opponent_afterstate, dtype=int8 )
                    states_buffer.append(opponent_afterstate)
                    i += 1
        m += 1

    return states_buffer, offsets, afterstates_dict

@njit(parallel=True)
def _select_optimal_move( values, offsets, afterstate_dict ):
    # receives an array of values for leaves of the 2-ply search and
    # selects the action which maximizes the expected value

    afterstates = list(afterstate_dict)
    move_expected_values = np.zeros( len(afterstates), np.float64 )
    
    for m in prange(len(afterstates)):
        d=0
        for r1 in range(1,7):
            for r2 in range(1,r1+1):
                p = 1/36 if r1==r2 else 2/36
                start_index = offsets[ m, d ]
                if d < NUM_SORTED_ROLLS - 1:
                    end_index = offsets[ m, d+1 ]
                else:
                    if m < len(afterstates) - 1:
                        end_index = offsets[ m+1, 0]
                    else:
                        end_index = len(values)
                minv = np.min( values[ start_index : end_index ] )
                move_expected_values[m] += p * minv
                d = d + 1

    return np.array(
        afterstate_dict[ afterstates[ np.argmax( move_expected_values ) ] ],
        dtype=int8 )

def _2_ply_search( state, player, dice, batch_value_function ):
    # this function calls into the numba code above from the python
    # interpreter, passes the array of states needing evaluation to a
    # provided batch_value_function (so that it can for example be
    # batch evaluated on a gpu), and then passes the resulting values
    # back to numba code. batch_value_function should return a numpy
    # or numba array, in particular it is batch_value_function's job
    # to convert, from say a JAX array, to an array numba can work
    # with

    state_buffer, offsets, player_moves = _collect_search_data( state,
                                                                player,
                                                                dice )

    value_buffer = batch_value_function( state_buffer )

    return _select_optimal_move( value_buffer, offsets, player_moves )

@njit(parallel=True)
def _vectorized_collect_search_data( state_vector, player_vector, dice_vector ):
    # vectorized version which takes arrays of states, players, and
    # dice, and returns a 1d buffer of states, an array of offset
    # matrices, and an array of state counts which together which
    # states belong to which combination of afterstates and dice.
    
    batch_size = len(state_vector)

    (state_buffer,
         offsets,
         player_moves) = _collect_search_data( state_vector[0],
                                               player_vector[0],
                                               dice_vector[0] )

    final_states_buffer_array = List()
    final_offsets = List()
    final_player_moves = List()

    # Preallocate the arrays so they can be written to in parallel
    for i in range(batch_size):
        final_states_buffer_array.append(state_buffer)
        final_offsets.append(offsets)
        final_player_moves.append(player_moves)

    for i in prange(batch_size):
        (state_buffer,
         offsets,
         player_moves) = _collect_search_data( state_vector[i],
                                               player_vector[i],
                                               dice_vector[i] )

        final_states_buffer_array[i] = state_buffer
        final_offsets[i] = offsets
        final_player_moves[i] = player_moves

    cumulative_state_counts = np.zeros( batch_size + 1, np.int64 )
    cumulative_state_count = 0
    for i in range(batch_size):
        cumulative_state_counts[i] = cumulative_state_count
        cumulative_state_count += len(final_states_buffer_array[i])
    cumulative_state_counts[batch_size] = cumulative_state_count

    final_states_buffer = List()
    for state_buffer in final_states_buffer_array:
        final_states_buffer.extend(state_buffer)

    return final_states_buffer, final_offsets, final_player_moves, cumulative_state_counts


@njit(parallel=True)
def _vectorized_select_optimal_move( final_values, final_offsets, final_player_moves,
                                     cumulative_state_counts ):

    batch_size = len(final_offsets)
    block_start = 0
    block_end = 0
    optimal_moves = np.empty( batch_size )
    
    for i in prange(batch_size):
        optimal_move = _select_optimal_move(
            final_values[ cumulative_state_counts[i]
                          : cumulative_state_counts[i+1]],
            final_offsets[i],
            final_player_moves[i] )
        optimal_moves[i] = optimal_move

    return optimal_moves

def _vectorized_2_ply_search( state_vector, player_vector, dice_vector, batch_value_function ):
    # this function calls into the numba code above from the python
    # interpreter, passes the returned array of states to a provided
    # batch_value_function (so that it can for example be batch
    # evaluated on a gpu, and then passes the resulting values back to
    # numba code. batch_value_function should return a numpy or numba
    # array, in particular it's batch_value_function's job to convert
    # from say a JAX array.

    (fin_state_buffer, fin_offsets, fin_player_moves,
     cum_state_counts) = _vectorized_collect_search_data( state, player, dice )

    fin_value_buffer = batch_value_function( fin_state_buffer )

    return _vectorized_select_optimal_move( fin_value_buffer,
                                            fin_offsets, fin_player_moves,
                                            cum_state_counts)
@njit(parallel=True)
def _linear_batch_value_function( feature_function, weights, states ):

    batch_size = len(states)
    
    values = np.empty( batch_size )
    
    for i in prange( batch_size ):
        values[i] = np.dot( weights, feature_function( states[i] ) )

    return values
