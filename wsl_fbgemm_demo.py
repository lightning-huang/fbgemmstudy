"""
Sumpool Mode 模拟 Non-Pool 执行演示
===================================
对比 Non-Pool 和 Sumpool 模拟两种方式，最终都输出 (batchSize, seq_len, emb) 作为下游输入

核心原理
--------
Non-Pool :  每个(batch, table)独立查表 → 小kernel碎片化 → GPU利用率低
Sumpool  :  所有indices concat后一次批量查表 → 通过offsets还原每条序列 → 结果等价但高效10~100x

场景设定
--------
  batch_size    = 80
  num_tables    = 8
  seq_len范围   = 1~50（每个batch每个feature随机）
  emb_dim       = 128
"""

import torch
import numpy as np

np.set_printoptions(precision=2, suppress=True, threshold=100, linewidth=200)
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 第1步：数据准备 — 生成变长 seq_len
# ============================================================================
print("=" * 80)
print("第1步：数据准备 — 变长 seq_len")
print("=" * 80)

BATCH_SIZE = 80
NUM_TABLES = 8
SEQ_LEN_MIN, SEQ_LEN_MAX = 1, 50
EMB_DIM = 128
NUM_EMBEDDINGS = 10000

# shape: (BATCH_SIZE, NUM_TABLES)
seq_lens = np.random.randint(SEQ_LEN_MIN, SEQ_LEN_MAX + 1, size=(BATCH_SIZE, NUM_TABLES))

print(f"batch_size={BATCH_SIZE}, num_tables={NUM_TABLES}, seq_len in [{SEQ_LEN_MIN},{SEQ_LEN_MAX}]")
print(f"seq_lens shape: {seq_lens.shape}")
print(f"\n前5个batch的seq_lens（每行代表一个batch_element,每列代表一个table）:")
print(seq_lens[:5])
print(f"\n各batch的总序列数: {seq_lens.sum(axis=1)[:5]}")

# ============================================================================
# 第2步：构建统一的 Embedding Tables
# ============================================================================
print("\n" + "=" * 80)
print("第2步：构建 Embedding Tables")
print("=" * 80)

# 8张表，每张表 10000个entry，emb_dim=128
tables = [torch.randn(NUM_EMBEDDINGS, EMB_DIM) for _ in range(NUM_TABLES)]
for i, t in enumerate(tables):
    tables[i] = (t - t.mean()) / (t.std() + 1e-8)  # 标准化，便于对比

print(f"构建了 {NUM_TABLES} 张表，每张表 shape: ({NUM_EMBEDDINGS}, {EMB_DIM})")

# ============================================================================
# 第3步：Non-Pool 执行 — 逐个(batch,table)独立查表
# ============================================================================
print("\n" + "=" * 80)
print("第3步：Non-Pool 执行 — 独立查表")
print("=" * 80)

def non_pool_lookup(tables, seq_lens):
    """
    Non-Pool 方式：对每个(batch, table)独立查表
    返回: dict[(b,t)] -> tensor (seq_len, emb_dim)
    """
    results = {}
    for t in range(NUM_TABLES):
        for b in range(BATCH_SIZE):
            seq_len = seq_lens[b, t]
            # 生成对应indices（每个batch元素的序列ID列表）
            indices = np.random.randint(0, NUM_EMBEDDINGS, size=seq_len)
            # 查表
            embs = tables[t][torch.IntTensor(indices)]
            results[(b, t)] = embs
    return results

print("  逐个(batch,table)查表...", end=" ", flush=True)
nonpool_results = non_pool_lookup(tables, seq_lens)
print("OK")
print(f"  产生 {BATCH_SIZE * NUM_TABLES} 个查表结果")

# 展示几个结果
b, t = 0, 0
print(f"\n  示例 non-pool 结果[(b={b},t={t})]: seq_len={seq_lens[b,t]}, emb:")
print(f"    前5维: {nonpool_results[(b,t)][0, :5].tolist()}")

# ============================================================================
# 第4步：Sumpool 模拟 Non-Pool — concat后批量查表
# ============================================================================
print("\n" + "=" * 80)
print("第4步：Sumpool 模拟 Non-Pool — concat + 批量查表 + 还原")
print("=" * 80)

def sumpool_simulate(tables, seq_lens):
    """
    Sumpool 模拟：
    1. 把所有(batch,table)的indices concat成一个长向量
    2. 按 table 顺序（table0所有batch → table1所有batch → ...）收集indices
    3. 一次性查所有表（通过每张表独立lookup然后concat）
    4. 通过 offsets 边界信息还原每条序列的embedding
    """
    # Step 1: 按batch遍历每个table，收集indices和offsets
    all_indices_concat = []   # 所有indices concat
    offsets = [0]            # 每条序列的起始位置（全局累积）
    table_ids_per_seq = []   # 记录每条序列属于哪张表
    batch_ids_per_seq = []   # 记录每条序列属于哪个batch

    for b in range(BATCH_SIZE):
        for t in range(NUM_TABLES):
            seq_len = seq_lens[b, t]
            indices = np.random.randint(0, NUM_EMBEDDINGS, size=seq_len)
            all_indices_concat.extend(indices.tolist())
            for _ in range(seq_len):
                table_ids_per_seq.append(t)
                batch_ids_per_seq.append(b)
            offsets.append(len(all_indices_concat))

    all_indices_concat = torch.IntTensor(all_indices_concat).cuda()
    offsets = torch.IntTensor(offsets).cuda()

    print(f"  [Step 1] concat后 indices总数: {len(all_indices_concat)}")
    print(f"  [Step 1] offsets数: {len(offsets)} (={BATCH_SIZE*NUM_TABLES}条序列+1)")
    print(f"  [Step 1] offsets[:5]: {offsets[:5].tolist()}")

    # Step 2: 按table分组查表（实际是按table维度展开）
    # sumpool的高明之处：把所有indices按table分段，每段查对应table
    # 然后按顺序还原每条序列
    # 这里用分组方式模拟（实际fbextemm内部通过table_offsets并行查）

    per_table_concat_indices = []
    for t in range(NUM_TABLES):
        t_indices = []
        for b in range(BATCH_SIZE):
            seq_len = seq_lens[b, t]
            indices = np.random.randint(0, NUM_EMBEDDINGS, size=seq_len)
            t_indices.extend(indices.tolist())
        per_table_concat_indices.append(torch.IntTensor(t_indices).cuda())

    # Step 3: 每张表一次查完（GPU高度并行）
    per_table_embs = [tables[t][per_table_concat_indices[t]] for t in range(NUM_TABLES)]
    print(f"  [Step 3] 每张表批量查表完成，embs shapes: {[e.shape for e in per_table_embs]}")

    # Step 4: 按 offsets 还原每条序列（un-pool / un-concat）
    # 对每条序列做 sum pool（与 non-pool 的语义等价）
    num_sequences = BATCH_SIZE * NUM_TABLES
    seq_embs = torch.zeros(num_sequences, EMB_DIM, device='cuda')

    for idx in range(num_sequences):
        start = offsets[idx]
        end = offsets[idx + 1]
        # 找出这条序列属于哪张表的哪个batch
        # 由于是按(b,t)顺序排列的，idx = b * NUM_TABLES + t
        t = idx % NUM_TABLES
        b = idx // NUM_TABLES
        # 在该表的那段连续indices中取对应位置
        # t表的indices区间：[b*NUM_TABLES+t 对应的全局位置]
        # 实际上per_table_concat_indices[t]包含table t所有batch的indices（按batch顺序）
        table_local_start = sum(seq_lens[b_, t] for b_ in range(b))
        table_local_end = table_local_start + seq_lens[b, t]
        table_emb = per_table_embs[t][table_local_start:table_local_end]
        seq_embs[idx] = table_emb.sum(dim=0)

    print(f"  [Step 4] sumpool还原完成，结果shape: {seq_embs.shape}")
    return seq_embs, offsets

print("  开始 sumpool 模拟...", end=" ", flush=True)
sumpool_embs, sumpool_offsets = sumpool_simulate(tables, seq_lens)
print("OK")

# ============================================================================
# 第5步：逐项对比 — 验证 sumpool == non-pool
# ============================================================================
print("\n" + "=" * 80)
print("第5步：逐项对比 — 验证结果一致")
print("=" * 80)

max_err = 0.0
samples_checked = 0

for t in range(NUM_TABLES):
    for b in range(BATCH_SIZE):
        # 获取 non-pool 结果
        np_emb = nonpool_results[(b, t)]  # (seq_len, emb_dim)
        np_sum = np_emb.sum(dim=0)         # sum pool 后 (emb_dim,)

        # 获取 sumpool 结果 — 找对应的 idx
        sumpool_idx = b * NUM_TABLES + t
        sp_emb = sumpool_embs[sumpool_idx]  # (emb_dim,)

        err = (np_sum - sp_emb).abs().max().item()
        max_err = max(max_err, err)
        samples_checked += 1

print(f"  逐项对比 {samples_checked} 个 (batch,table) 组合")
print(f"  最大绝对误差: {max_err:.2e}")
print(f"  结论: {'✅ 结果完全一致（误差在浮点精度内）' if max_err < 1e-5 else '❌ 结果不一致'}")

# ============================================================================
# 第6步：最终输出形状 — 两种方式都输出 (batchSize, seq_len, emb)
# ============================================================================
print("\n" + "=" * 80)
print("第6步：最终输出形状")
print("=" * 80)

print("""
 Non-Pool 最终输出:
   {b,t} -> sum_pool -> (emb_dim,)       每个(batch,table)一个向量
   reshape成: (BATCH_SIZE, NUM_TABLES, EMB_DIM) = (80, 8, 128)
   下一步用: seq_lens[b,t] 标记每条序列的实际长度

 Sumpool 最终输出:
   sumpool_embs shape: (BATCH_SIZE * NUM_TABLES, EMB_DIM) = (640, 128)
   reshape成: (BATCH_SIZE, NUM_TABLES, EMB_DIM) = (80, 8, 128)
   与 non-pool 完全相同的最终形状

 给下一步使用时的接口:
   output = sumpool_embs.reshape(BATCH_SIZE, NUM_TABLES, EMB_DIM)
   下游网络按 seq_lens[b,t] 取每条序列的有效部分
""")

# 具体演示 reshape
final_output = sumpool_embs.reshape(BATCH_SIZE, NUM_TABLES, EMB_DIM)
print(f"  最终 output shape: {final_output.shape}")
print(f"  output[b, t, :] = 第b个batch第t张表的embedding向量 (128,)")
print(f"\n  非池化版本（非sum）：")
print(f"    需要输出 (total_sequences, max_seq_len, EMB_DIM)")
print(f"    total_sequences = {seq_lens.sum()}")
print(f"    max_seq_len = {seq_lens.max()}")
print(f"    即 ({seq_lens.sum()}, {seq_lens.max()}, {EMB_DIM})")

# ============================================================================
# 第7步：性能对比（计算量估算）
# ============================================================================
print("\n" + "=" * 80)
print("第7步：效率对比")
print("=" * 80)

import time

# Non-pool timing
start = time.perf_counter()
for _ in range(10):  # 跑10次取平均
    _ = non_pool_lookup(tables, seq_lens)
nonpool_time = (time.perf_counter() - start) / 10

# Sumpool timing
start = time.perf_counter()
for _ in range(10):
    _ = sumpool_simulate(tables, seq_lens)
sumpool_time = (time.perf_counter() - start) / 10

print(f"  Non-Pool 每次耗时: {nonpool_time*1000:.2f} ms")
print(f"  Sumpool  每次耗时: {sumpool_time*1000:.2f} ms")
print(f"  加速比: {nonpool_time/sumpool_time:.1f}x")
print(f"\n  注: 本demo在CPU上运行，GPU上差距更显著（10~100x）")
print(f"  原因: Non-pool产生{BATCH_SIZE*NUM_TABLES}次独立小kernel，")
print(f"        Sumpool只产生{NUM_TABLES}次批量查表")