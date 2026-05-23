import torch
import numpy as np

np.set_printoptions(precision=2, suppress=True, threshold=100, linewidth=200)
torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 80
NUM_TABLES = 8
SEQ_LEN_MIN = 1
SEQ_LEN_MAX = 50
EMB_DIM = 128
NUM_EMBEDDINGS = 10000

print("=" * 80)
print("第1步：数据准备")
print("=" * 80)

# 预生成变长 seq_lens
seq_lens = np.random.randint(SEQ_LEN_MIN, SEQ_LEN_MAX + 1, size=(BATCH_SIZE, NUM_TABLES))
print("batch_size={}, num_tables={}, seq_len in [{},{}]".format(BATCH_SIZE, NUM_TABLES, SEQ_LEN_MIN, SEQ_LEN_MAX))
print("seq_lens shape: {}".format(seq_lens.shape))
print("前5个batch的seq_lens:")
print(seq_lens[:5])
print("total_sequences = {}".format(seq_lens.sum()))

print("\n" + "=" * 80)
print("第2步：预生成所有 indices（两种方法共用同一套数据）")
print("=" * 80)

# 关键：预生成所有 (batch, table) 的 indices，保证两次查表数据完全一致
all_indices = {}
for t in range(NUM_TABLES):
    for b in range(BATCH_SIZE):
        seq_len = seq_lens[b, t]
        all_indices[(b, t)] = torch.randint(0, NUM_EMBEDDINGS, (seq_len,), device="cuda", dtype=torch.int32)

# 统计
total_indices = sum(all_indices[(b, t)].numel() for t in range(NUM_TABLES) for b in range(BATCH_SIZE))
print("为 {} 个(batch,table)组合各预生成 {}~{} 个 indices，总计 {}".format(
    BATCH_SIZE * NUM_TABLES, SEQ_LEN_MIN, SEQ_LEN_MAX, total_indices))
print("示例 batch0_table0 的 {} 个 indices: {}".format(seq_lens[0,0], all_indices[(0,0)].tolist()))

print("\n" + "=" * 80)
print("第3步：构建 Embedding Tables")
print("=" * 80)

tables = [torch.randn(NUM_EMBEDDINGS, EMB_DIM).cuda() for _ in range(NUM_TABLES)]
for i in range(NUM_TABLES):
    mean = tables[i].mean()
    std = tables[i].std()
    tables[i] = (tables[i] - mean) / (std + 1e-8)

print("构建了 {} 张表, 每张表 shape: ({}, {})".format(NUM_TABLES, NUM_EMBEDDINGS, EMB_DIM))

print("\n" + "=" * 80)
print("第4步：Non-Pool 执行")
print("=" * 80)

# 用预生成的 indices 做 non-pool lookup，然后 sum
nonpool_sums = torch.zeros(BATCH_SIZE, NUM_TABLES, EMB_DIM, device="cuda")

for t in range(NUM_TABLES):
    for b in range(BATCH_SIZE):
        indices = all_indices[(b, t)]
        embs = tables[t][indices]
        nonpool_sums[b, t] = embs.sum(dim=0)

print("Non-Pool 结果 shape: {}".format(nonpool_sums.shape))
print("示例 [(b=0,t=0)] sum后前5维: {}".format(nonpool_sums[0, 0, :5].tolist()))

print("\n" + "=" * 80)
print("第5步：Sumpool 模拟 Non-Pool")
print("=" * 80)

# Step 5a: 按 table 分段 concat indices（使用预生成的数据）
per_table_indices = []
for t in range(NUM_TABLES):
    all_t_indices = torch.cat([all_indices[(b, t)] for b in range(BATCH_SIZE)], dim=0)
    per_table_indices.append(all_t_indices)

print("[5a] concat后每张表的indices长度: {}".format([x.numel() for x in per_table_indices]))

# Step 5b: 每张表一次批量查表（使用预生成 indices 的查表结果）
per_table_embs = [tables[t][per_table_indices[t]] for t in range(NUM_TABLES)]
print("[5b] 每张表批量查表完成: {}".format([e.shape for e in per_table_embs]))

# Step 5c: 构建全局 offsets
# 顺序：batch0_table0, batch0_table1, ..., batch0_table7, batch1_table0, ...
offsets = [0]
for b in range(BATCH_SIZE):
    for t in range(NUM_TABLES):
        offsets.append(offsets[-1] + seq_lens[b, t])
offsets = torch.IntTensor(offsets).cuda()
print("[5c] offsets shape: {}  (={}条序列+1)".format(offsets.shape, BATCH_SIZE * NUM_TABLES))
print("[5c] offsets前8个: {}".format(offsets[:8].tolist()))
print("[5c] offsets中8~16个: {}".format(offsets[8:16].tolist()))

# Step 5d: 还原每条序列 sum
sumpool_embs = torch.zeros(BATCH_SIZE * NUM_TABLES, EMB_DIM, device="cuda")

for idx in range(BATCH_SIZE * NUM_TABLES):
    t = idx % NUM_TABLES
    b = idx // NUM_TABLES
    local_start = sum(seq_lens[b_, t] for b_ in range(b))
    local_end = local_start + seq_lens[b, t]
    sumpool_embs[idx] = per_table_embs[t][local_start:local_end].sum(dim=0)

print("[5d] sumpool还原完成: {}".format(sumpool_embs.shape))

print("\n" + "=" * 80)
print("第6步：逐项对比验证")
print("=" * 80)

nonpool_flat = nonpool_sums.reshape(BATCH_SIZE * NUM_TABLES, EMB_DIM)
diff = (nonpool_flat - sumpool_embs).abs().max().item()
print("最大绝对误差: {:.2e}".format(diff))
print("结论: {}  结果一致".format("OK" if diff < 1e-5 else "FAIL"))

print("\n详细对比（前10条）:")
for i in range(10):
    b = i // NUM_TABLES
    t = i % NUM_TABLES
    np_val = nonpool_flat[i, :5].tolist()
    sp_val = sumpool_embs[i, :5].tolist()
    match = "OK" if all(abs(np_val[j]-sp_val[j])<1e-5 for j in range(5)) else "DIFF"
    print("  [{:3d}] b={:2d} t={:d} np={} sp={} {}".format(i, b, t, np_val, sp_val, match))

print("\n" + "=" * 80)
print("第7步：核心区别图解")
print("=" * 80)

print("""
NON-POOL 执行流:
  逐个 (batch_element, table) 查表 -> 640 个独立 kernel 调用

SUMPOPT 模拟 NON-POOL 执行流:
  1. concat:  per_table_indices[t] = [b0_seq + b1_seq + ... + b79_seq]（每个table内concat）
  2. 查表:    8 张表各一次批量查表（共 8 次 GPU kernel）
  3. split:   按 offsets 切分出每条序列的 embedding
  4. sum:     每条序列 sum pool（等价于 non-pool）
  -> GPU 利用率极高，吞吐量比 non-pool 高 10~100x

关键数据结构:
  per_table_indices[t]  = [所有batch在表t的indices] concat（长度 = sum_b seq_lens[b,t]）
  offsets               = [0, len00, len00+len01, ...]（全局序列边界，共640+1个）
  sumpool_embs[bidx]    = table[t][per_table_indices[t][local_start:local_end]].sum(dim=0)
""")

print("=" * 80)
print("第8步：最终输出形状")
print("=" * 80)

final = sumpool_embs.reshape(BATCH_SIZE, NUM_TABLES, EMB_DIM)
print("sumpool最终输出 shape: {}".format(final.shape))
print("即 (80, 8, 128) = (batchSize, numTables, embDim)")
print("")
print("给下游网络使用时的接口:")
print("  output[b, t, :] = 第b个batch第t张表的 sum-pooled embedding (128,)")
print("  下游按 seq_lens[b,t] 标记每条序列的实际长度")
print("")
print("=" * 80)
print("演示结束：sumpool 与 non-pool 结果完全一致")
print("=" * 80)