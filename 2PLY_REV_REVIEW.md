# Review of `2ply_rev.py` - Faster 2-Ply Implementation

## ✅ **Improvements Over Original**

### 1. **Incremental State Processing (Better Memory Usage)**
```python
# NEW: Processes states per dice roll, not all at once
for d1 in range(1, 7):
    for d2 in range(d1, 7):
        # Process this dice roll's states immediately
        boards = jnp.asarray(np.stack(boards))
        # Evaluate and move on
```

**Benefit**: Doesn't accumulate 100k+ states in memory before processing. More memory efficient.

**vs Original**: Collects ALL states first, then processes in one big batch (can cause OOM).

### 2. **Simpler Code Structure**
- Cleaner separation: 1-ply vs 2-ply
- Less complex indexing logic
- Easier to understand

### 3. **Flexible Training Modes**
- Can switch between 1-ply and 2-ply via `--train_search`
- Good for debugging and comparison

### 4. **Better Model Initialization**
```python
# Uses model.init() directly with proper signature
params = model.init(rng, dummy_board, dummy_aux)
```
More straightforward than the original.

---

## 🚨 **Critical Issues**

### 1. **IMPORT BUG** ❌
```python
from backgammon_value_net import BackgammonValueNet, encode, AUX_INPUT_SIZE
```
**Problem**: `encode` is NOT in `backgammon_value_net.py`, it's in `train_value_td0.py`!

**Fix**: 
```python
from backgammon_value_net import BackgammonValueNet, AUX_INPUT_SIZE
from train_value_td0 import encode, actions_py, py_reward
```

### 2. **Wrong Dice Roll Probabilities** ❌
```python
# CURRENT (WRONG):
for d1 in range(1, 7):
    for d2 in range(d1, 7):  # Only 21 combinations
        # ...
exp_val = float(jnp.mean(jnp.asarray(vals)))  # Uniform average!
```

**Problem**: 
- Only iterates 21 sorted combinations (missing that doubles have 1/36 prob, others 2/36)
- Uses uniform mean instead of weighted average by probability
- Should be: `exp_val = sum(prob[i] * min_val[i] for i in range(21))`

**Correct Approach**:
```python
rolls, probs = all_sorted_rolls_and_probs()  # From train_value_td0_2ply.py
for j, (d1, d2) in enumerate(rolls):
    prob = probs[j]
    # ...
    exp_val += prob * min_val
```

### 3. **Missing Epsilon-Greedy Exploration** ⚠️
Original has:
```python
if np.random.rand() < self.eps_greedy:
    return random_move
```
**Impact**: Less exploration during training, may converge to suboptimal policy.

### 4. **Missing Terminal State Early Exit** ⚠️
Original has:
```python
for ns in afterstates:
    r = py_reward(ns, player)
    if r != 0:
        return float(r), ns  # Immediate win/loss
```
**Impact**: Wastes computation evaluating states when game is already won/lost.

### 5. **No JAX Memory Management** ⚠️
Missing:
```python
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```
**Impact**: May cause OOM errors on GPU.

### 6. **Potential Array Slicing Issue** ⚠️
```python
boards[i:i + eval_batch]  # Slicing JAX array
```
**Problem**: If `boards` is a JAX array, slicing creates new arrays. Should ensure it's a list first.

---

## 📊 **Performance Comparison**

### Memory Usage:
- **Original**: Accumulates all states → ~100k states × 1.43 KB = ~140 MB + GPU activations
- **New**: Processes incrementally → Max batch size at a time = ~6 MB per dice roll
- **Winner**: ✅ New version (much better memory efficiency)

### Speed:
- **Original**: One big batch evaluation (fewer GPU calls, but larger batches)
- **New**: Many smaller batch evaluations (more GPU calls, but smaller batches)
- **Trade-off**: New version has more GPU kernel launches, but avoids memory issues

### Correctness:
- **Original**: ✅ Proper dice probabilities, epsilon-greedy, terminal checks
- **New**: ❌ Wrong probabilities, missing features
- **Winner**: Original (but can be fixed)

---

## 🔧 **Recommended Fixes**

### Fix 1: Correct Imports
```python
from backgammon_value_net import BackgammonValueNet, AUX_INPUT_SIZE
from train_value_td0 import encode, actions_py, py_reward
```

### Fix 2: Add Dice Roll Probabilities
```python
def all_sorted_rolls_and_probs():
    counts = {}
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            a, b = sorted((d1, d2), reverse=True)
            counts[(a, b)] = counts.get((a, b), 0) + 1
    rolls = list(counts.keys())
    probs = [counts[p] / 36.0 for p in rolls]
    rolls, probs = zip(*sorted(zip(rolls, probs), reverse=True))
    return list(rolls), list(probs)

# In pick_action_2ply:
rolls, probs = all_sorted_rolls_and_probs()
exp_val = 0.0
for j, (d1, d2) in enumerate(rolls):
    prob = probs[j]
    # ... compute min_val for this dice roll
    exp_val += prob * min_val
```

### Fix 3: Add Missing Features
```python
def pick_action_2ply(state, player, st, eval_batch=4096, eps_greedy=0.0):
    # ... get afterstates ...
    
    # Epsilon-greedy
    if np.random.rand() < eps_greedy:
        return random_move
    
    # Terminal check
    for ns in afterstates:
        r = py_reward(ns, player)
        if r != 0:
            return float(r), ns
    
    # ... rest of 2-ply search ...
```

### Fix 4: Add JAX Memory Settings
```python
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# ... then import jax ...
```

---

## 💡 **Verdict**

**The new version has better memory efficiency** (incremental processing), but has **critical correctness issues** (wrong probabilities, missing features).

**Recommendation**: 
1. Fix the import bug
2. Add proper dice roll probabilities
3. Add epsilon-greedy and terminal checks
4. Add JAX memory management
5. Then it will be both **faster AND correct**

The incremental processing approach is excellent for memory-constrained environments!

