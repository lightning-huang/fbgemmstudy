# Sumpool vs Non-Pool on TBE (IntNBitTableBatchedEmbeddingBagsCodegen)

---

**Goal**: Produce `(T, B, S_max, D)` — one embedding **per index** (no sum),
with all tables aligned to the longest sequence length.

**Two equivalent paths**:
- `PoolingMode.SUM` + per-row offsets → one sum per index → effectively no pooling
- `PoolingMode.NONE` + same offsets → raw per-index embeddings

| | `PoolingMode.SUM` (sumpool) | `PoolingMode.NONE` (non-pool) |
|---|---|---|
| forward output | `(T, B*S_max, D)` — one emb per index | `(T, B*S_max, D)` — same |
| to `(T, B, S_max, D)` | reshape | reshape |
| offsets shape | `(T * B * S_max + 1,)` | same |
| indices | same `concat_all` | same |
| padding | dedicated PAD_IDX, masked after | same |

**Key**: Both modes are mathematically identical when each bag has exactly
one index. We just reshape `(T, B*S_max, D)` → `(T, B, S_max, D)` and
mask out padding positions.

---

## Step 0: Config & Constants


```python
import torch
import numpy as np
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    PoolingMode,
    EmbeddingLocation,
    BoundsCheckMode,
)
from fbgemm_gpu.split_embedding_configs import SparseType

np.set_printoptions(precision=4, suppress=True, threshold=100, linewidth=200)
torch.manual_seed(42)
np.random.seed(42)

B = 10          # batch size
T = 8           # table count
D = 128         # embedding dimension
V = 10000       # vocab size
PAD_IDX = 0     # dedicated padding index (guaranteed out-of-vocab)
SEQ_LEN_MIN, SEQ_LEN_MAX = 2, 50

print("Config: B={}, T={}, D={}, V={}, seq=[{},{}]".format(
    B, T, D, V, SEQ_LEN_MIN, SEQ_LEN_MAX))
```

## Step 1: Generate Variable-Length Sequences

`seq_lens[b, t]` = actual length for batch `b`, table `t`.
`S_max = max(seq_lens)` — all tables align to this.


```python
# seq_lens[b, t]
seq_lens = np.random.randint(SEQ_LEN_MIN, SEQ_LEN_MAX + 1, size=(B, T))
S_max = int(seq_lens.max())
print("seq_lens shape: (B={}, T={})".format(B, T))
print("seq_lens (first 5 batches):\n{}".format(seq_lens[:5]))
print("S_max = {} (all tables pad to this)".format(S_max))
```

## Step 2: Build concat_all + offsets for Per-Index Bags

Each index in a sequence → its own bag.

For table `t`, batch `b`:
  - indices for valid positions `[0, seq_lens[b,t])` → real embeddings
  - indices for padding positions `[seq_lens[b,t], S_max)` → PAD_IDX (masked later)

`concat_all[t, b, s]` = index at position `s` for batch `b`, table `t`.

**offsets layout** (per table, then flattened):
`offsets[t, b*S_max + s]` = cumulative count of indices up to bag `(t,b,s)`

Result: `concat_all` shape `(T, B*S_max,)` → flattened to `(T*B*S_max,)`
         `offsets` shape `(T*B*S_max + 1,)`


```python
# Build per-table (T, B*S_max) index tensors, then flatten
concat_per_table = []   # list of (B*S_max,) tensors, one per table

for t in range(T):
    table_indices = []
    for b in range(B):
        seq_len = seq_lens[b, t]
        # real indices
        real_idx = torch.randint(1, V, (seq_len,), dtype=torch.int32, device="cuda")
        # padding indices
        pad_idx = torch.full((S_max - seq_len,), PAD_IDX, dtype=torch.int32, device="cuda")
        table_indices.append(torch.cat([real_idx, pad_idx], dim=0))
    concat_t = torch.cat(table_indices, dim=0)   # (B*S_max,)
    concat_per_table.append(concat_t)

# concat_all: (T, B*S_max,) — per-table index blocks
concat_all = torch.cat(concat_per_table, dim=0)  # (T*B*S_max,)
print("concat_all shape: {} = (T*B*S_max,)".format(concat_all.shape))

# Offsets: cumulative count of indices seen so far
# Each bag has exactly 1 index → offsets[i] = i
offsets = torch.arange(0, T * B * S_max + 1, dtype=torch.int32, device="cuda")
print("offsets shape: {} = (T*B*S_max+1,)".format(offsets.shape))
print("offsets[:8]:  {}".format(offsets[:8].tolist()))
print("offsets[-3:]:  {}".format(offsets[-3:].tolist()))
```

## Step 3: SUM Mode — One Bag Per Index (Effectively No Pooling)


```python
embedding_specs = [
    ("t{}".format(t), V, D, SparseType.FP16, EmbeddingLocation.DEVICE)
    for t in range(T)
]

tb_sum = IntNBitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=embedding_specs,
    pooling_mode=PoolingMode.SUM,
    device="cuda",
    bounds_check_mode=BoundsCheckMode.WARNING,
)
tb_sum.initialize_weights()
tb_sum.initialize_logical_weights_placements_and_offsets()
tb_sum.initialize_physical_weights_placements_and_offsets()

# SUM forward: one bag per index → (T*B*S_max, D)
sum_out = tb_sum.forward(concat_all, offsets)
print("SUM raw output: {} = (T*B*S_max, D)".format(sum_out.shape))

# Reshape → (T, B, S_max, D)
sum_reshaped = sum_out.reshape(T, B, S_max, D)
print("SUM reshaped : {} = (T, B, S_max, D)".format(sum_reshaped.shape))
```

## Step 4: Mask Padding Positions

Wherever we put PAD_IDX (index 0), the output embedding is garbage —
zero it out so downstream sees a proper `(T, B, S_max, D)` tensor.


```python
# Build mask: (T, B, S_max) — True = real, False = padding
mask = torch.zeros(T, B, S_max, dtype=torch.bool, device="cuda")
for t in range(T):
    for b in range(B):
        mask[t, b, :seq_lens[b, t]] = True

# Zero out padding positions
sum_final = sum_reshaped.clone()
sum_final = sum_final * mask.unsqueeze(-1)    # (T, B, S_max, D) * (T,B,S_max,1)
print("Masked sum_final: {} — padding zeros = {}".format(
    sum_final.shape, (~mask).sum().item()))
```

## Step 5: NONE Mode — Same Result


```python
tb_none = IntNBitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=embedding_specs,
    pooling_mode=PoolingMode.NONE,
    device="cuda",
    bounds_check_mode=BoundsCheckMode.WARNING,
)
tb_none.initialize_weights()
tb_none.initialize_logical_weights_placements_and_offsets()
tb_none.initialize_physical_weights_placements_and_offsets()

# NONE: same inputs → same (T*B*S_max, D) raw output
none_out = tb_none.forward(concat_all, offsets)
print("NONE raw output: {} = (T*B*S_max, D)".format(none_out.shape))

# Reshape + mask — identical result
none_reshaped = none_out.reshape(T, B, S_max, D)
none_final = none_reshaped * mask.unsqueeze(-1)
print("NONE final    : {} = (T, B, S_max, D)".format(none_final.shape))
```

## Step 6: Verification — SUM vs NONE


```python
diff_abs = (sum_final - none_final).abs().max().item()
diff_rel = ((sum_final - none_final) / (sum_final.abs() + 1e-8)).abs().max().item()

print("Max absolute error: {:.6e}".format(diff_abs))
print("Max relative error: {:.6e}".format(diff_rel))
print("Match: {}".format("✅ OK" if diff_abs < 1e-4 else "❌ FAIL"))
```

## Step 7: Calling Pattern Summary


| | `PoolingMode.SUM` (sumpool) | `PoolingMode.NONE` (non-pool) |
|---|---|---|
| `forward(concat_all, offsets)` | ✅ same call | ✅ same call |
| offsets format | `(T*B*S_max + 1,)` = arange | `(T*B*S_max + 1,)` = arange |
| concat_all format | `(T*B*S_max,)` — real+pad | same |
| raw output | `(T*B*S_max, D)` | `(T*B*S_max, D)` |
| reshape → `(T, B, S_max, D)` | reshape | reshape |
| mask padding | `mask.unsqueeze(-1) * out` | same |
| internal GPU work | one lookup + 1×1 sum | one lookup, no sum |
| GPU kernel count | T (one gemm/table) | T (same gemm) |

**Formula**:
```text
# 1. Build indices with padding
seq_lens[b,t] ∈ [min_len, max_len]
S_max = seq_lens.max()
concat_all = [real_indices, PAD_IDX × (S_max − seq_len)] per (b,t)

# 2. Offsets: each bag = 1 index
offsets = arange(0, T*B*S_max + 1, dtype=torch.int32)

# 3. Forward (SUM or NONE — same result)
raw = tb.forward(concat_all, offsets)          # (T*B*S_max, D)

# 4. Reshape + mask
out = raw.reshape(T, B, S_max, D)             # (T, B, S_max, D)
mask[b,t,s] = (s < seq_lens[b,t])
out = out * mask.unsqueeze(-1)                 # zero padding positions
```


```python
print("""
FINAL RESULT:
  Shape  : (T={}, B={}, S_max={}, D={})
  Config : {} tables, batch {}, max_seq_len {}, emb_dim {}
  Padding: {} positions masked to zero
  SUM vs NONE max abs error: {:.6e}
""".format(
    T, B, S_max, D,
    T, B, S_max, D,
    (~mask).sum().item(),
    diff_abs,
))
```
